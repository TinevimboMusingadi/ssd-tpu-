"""LLM inference engine: AR, SD, SSD, Instance modes."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Iterator

import jax.numpy as jnp

from connect.mesh_allocator import allocate_meshes, mesh_policy_from_env
from jax_ssd.algorithm.async_protocol import DraftCommand, DraftRequest, VerifyOutcome
from jax_ssd.algorithm.instance_prior import (
    find_high_salience_spans,
    score_token_salience,
    spans_to_speculation_tokens,
)
from jax_ssd.algorithm.verify import verify_greedy
from jax_ssd.config import DecodeMode, SSDConfig
from jax_ssd.models.base import DecodeModelAdapter
from jax_ssd.models.model_loader import load_model_adapter
from jax_ssd.runtime.metrics import MetricsCollector, RunMetrics, StepMetrics
from jax_ssd.runtime.page_manager import PageManager
from jax_ssd.runtime.scheduler import Scheduler
from jax_ssd.runtime.sequence import Sequence
from jax_ssd.runtime.workers import DraftWorker
from jax_ssd.sampling_params import SamplingParams


TokenCallback = Callable[[int, str], None]


class LLMEngine:
    def __init__(self, config: SSDConfig) -> None:
        if config.mesh_allocation is None:
            from dataclasses import replace

            policy = mesh_policy_from_env()
            if config.mode == DecodeMode.AR:
                policy = "all_target"
            elif config.tpu_role == "target":
                policy = "all_target"
            elif config.tpu_role == "draft":
                policy = "all_draft"
            config = replace(config, mesh_allocation=allocate_meshes(policy=policy))
        self.config = config
        role = config.tpu_role
        self.page_manager = PageManager(
            block_size=config.block_size,
            num_blocks=config.num_blocks,
        )
        self.scheduler = Scheduler(config, self.page_manager)
        alloc = config.mesh_allocation
        share_key = None
        if (
            not config.use_toy_model
            and config.target_model_path
            and config.target_model_path == config.draft_model_path
        ):
            share_key = f"shared:{config.target_model_path}"
        load_target = role in ("both", "target")
        # AR only needs the target model; skip draft Gemma4 load (assistant weights optional).
        load_draft = role in ("both", "draft") and config.mode != DecodeMode.AR

        if load_target:
            self.target = load_model_adapter(
                config.target_model_path,
                use_toy=config.use_toy_model,
                share_key=share_key,
                mesh=alloc.target_mesh if alloc else None,
                devices=alloc.target_devices if alloc else None,
                role="target",
            )
        else:
            from jax_ssd.models.toy_model import ToyModelAdapter

            self.target = ToyModelAdapter(seed=99)

        if load_draft:
            self.draft = load_model_adapter(
                config.draft_model_path,
                use_toy=config.use_toy_model,
                seed=1,
                share_key=share_key,
                mesh=alloc.draft_mesh if alloc else None,
                devices=alloc.draft_devices if alloc else None,
                role="draft",
            )
        else:
            from jax_ssd.models.toy_model import ToyModelAdapter

            self.draft = ToyModelAdapter(seed=1)

        self.draft_worker: DraftWorker | None = None
        if load_draft and config.mode in (DecodeMode.SSD, DecodeMode.INSTANCE):
            self.draft_worker = DraftWorker(config, self.draft)
            self.draft_worker.start()

    def generate(
        self,
        prompts: list[list[int]],
        sampling: SamplingParams,
        *,
        on_token: TokenCallback | None = None,
    ) -> tuple[list[list[int]], RunMetrics]:
        collector = MetricsCollector(mode=self.config.mode.value)
        collector.start()
        outputs: list[list[int]] = []

        for prompt in prompts:
            seq = self.scheduler.add_request(prompt)
            self.scheduler.schedule()
            if self.draft_worker and self.config.mode == DecodeMode.INSTANCE:
                self.draft_worker.set_context(prompt)

            if self.config.mode == DecodeMode.AR:
                tokens = self._run_ar(seq, sampling, on_token, collector)
            elif self.config.mode == DecodeMode.SD:
                tokens = self._run_sd(seq, sampling, on_token, collector)
            elif self.config.mode in (DecodeMode.SSD, DecodeMode.INSTANCE):
                tokens = self._run_ssd(seq, sampling, on_token, collector)
            else:
                tokens = self._run_ar(seq, sampling, on_token, collector)

            outputs.append(tokens)

        metrics = collector.finish()
        return outputs, metrics

    def generate_stream(
        self,
        prompt: list[int],
        sampling: SamplingParams,
        on_token: TokenCallback | None = None,
    ) -> Iterator[tuple[int, str]]:
        """Yield (token_id, decoded_str) as tokens are generated."""
        collected: list[int] = []

        def _cb(tok: int, s: str) -> None:
            collected.append(tok)
            if on_token:
                on_token(tok, s)

        self.generate([prompt], sampling, on_token=_cb)
        for tok in collected:
            yield tok, self.target.decode_tokens_to_str([tok])

    def _decode_callback(
        self,
        model: DecodeModelAdapter,
        token: int,
        on_token: TokenCallback | None,
    ) -> None:
        if on_token:
            on_token(token, model.decode_tokens_to_str([token]))

    def _commit(self, model: DecodeModelAdapter, token_ids: list[int]) -> None:
        commit = getattr(model, "commit_tokens", None)
        if commit is not None:
            commit(token_ids)

    def _run_ar(
        self,
        seq: Sequence,
        sampling: SamplingParams,
        on_token: TokenCallback | None,
        collector: MetricsCollector,
    ) -> list[int]:
        gen_ar = getattr(self.target, "generate_ar", None)
        if gen_ar is not None:
            t0 = time.perf_counter()
            generated = gen_ar(
                list(seq.prompt_token_ids),
                sampling.max_new_tokens,
                on_token,
            )
            if generated:
                collector.record_first_token()
            for tok in generated:
                seq.append_token(tok)
            collector.record_step(
                StepMetrics(
                    target_verify_ms=(time.perf_counter() - t0) * 1000,
                    accepted_tokens=len(generated),
                )
            )
            return seq.completion_token_ids

        kv = self.target.allocate_kv()
        page_table = jnp.zeros((self.config.num_blocks,), dtype=jnp.int32)
        pre = self.target.prefill(jnp.array(seq.prompt_token_ids), kv, page_table)
        logits = pre.logits[-1]
        token = int(jnp.argmax(logits))
        seq.append_token(token)
        self._decode_callback(self.target, token, on_token)
        collector.record_first_token()

        while not seq.is_finished(None, sampling.max_new_tokens):
            t0 = time.perf_counter()
            dec = self.target.decode(token, seq.num_tokens, kv, page_table)
            token = dec.next_token
            seq.append_token(token)
            self._decode_callback(self.target, token, on_token)
            collector.record_step(
                StepMetrics(
                    target_verify_ms=(time.perf_counter() - t0) * 1000,
                    accepted_tokens=1,
                )
            )

        return seq.completion_token_ids

    def _run_sd(
        self,
        seq: Sequence,
        sampling: SamplingParams,
        on_token: TokenCallback | None,
        collector: MetricsCollector,
    ) -> list[int]:
        k = self.config.speculate_k
        kv = self.target.allocate_kv()
        page_table = jnp.zeros((self.config.num_blocks,), dtype=jnp.int32)
        pre = self.target.prefill(jnp.array(seq.prompt_token_ids), kv, page_table)
        recovery = int(jnp.argmax(pre.logits[-1]))
        seq.recovery_token = recovery
        seq.append_token(recovery)
        self._decode_callback(self.target, recovery, on_token)
        collector.record_first_token()

        while not seq.is_finished(None, sampling.max_new_tokens):
            t0 = time.perf_counter()
            spec, _ = self.draft.draft_speculate(seq.token_ids, k)
            spec_kp1 = jnp.array([recovery] + spec, dtype=jnp.int32)
            vres = self.target.verify(spec_kp1, jnp.arange(k + 1), kv, page_table)
            vout = verify_greedy(vres.logits, spec_kp1[None, :])
            acc = int(vout.accept_until[0])
            accepted = spec[:acc]
            recovery = int(vout.recovery_tokens[0])

            for tok in accepted:
                seq.append_token(tok)
                self._decode_callback(self.target, tok, on_token)
            seq.append_token(recovery)
            self._decode_callback(self.target, recovery, on_token)
            self._commit(self.target, accepted + [recovery])

            collector.record_step(
                StepMetrics(
                    target_verify_ms=(time.perf_counter() - t0) * 1000,
                    accepted_tokens=len(accepted) + 1,
                )
            )

        return seq.completion_token_ids

    def _run_ssd(
        self,
        seq: Sequence,
        sampling: SamplingParams,
        on_token: TokenCallback | None,
        collector: MetricsCollector,
    ) -> list[int]:
        assert self.draft_worker is not None
        k = self.config.speculate_k
        kv = self.target.allocate_kv()
        page_table = jnp.zeros((self.config.num_blocks,), dtype=jnp.int32)
        pre = self.target.prefill(jnp.array(seq.prompt_token_ids), kv, page_table)
        recovery = int(jnp.argmax(pre.logits[-1]))
        seq.recovery_token = recovery
        seq.append_token(recovery)
        self._decode_callback(self.target, recovery, on_token)
        collector.record_first_token()

        # Initial draft request
        key = jnp.array([[seq.seq_id, 0, recovery]], dtype=jnp.int32)
        self.draft_worker.submit(
            DraftRequest(
                command=DraftCommand.SERVE,
                cache_keys=key,
                seq_lens=jnp.array([seq.num_tokens], dtype=jnp.int32),
                page_tables=page_table[None, :],
                temperatures=jnp.array([0.0]),
            )
        )
        draft_resp = self.draft_worker.get_response()

        while not seq.is_finished(None, sampling.max_new_tokens):
            t0 = time.perf_counter()
            spec = [int(t) for t in draft_resp.spec_tokens[0]]
            spec_kp1 = jnp.array([recovery] + spec, dtype=jnp.int32)
            vres = self.target.verify(spec_kp1, jnp.arange(k + 1), kv, page_table)
            vout = verify_greedy(vres.logits, spec_kp1[None, :])
            acc = int(vout.accept_until[0])
            accepted = spec[:acc]
            recovery = int(vout.recovery_tokens[0])
            cache_hit = bool(draft_resp.cache_hit[0])

            for tok in accepted:
                seq.append_token(tok)
                self._decode_callback(self.target, tok, on_token)
            seq.append_token(recovery)
            self._decode_callback(self.target, recovery, on_token)
            self._commit(self.target, accepted + [recovery])

            collector.record_step(
                StepMetrics(
                    target_verify_ms=(time.perf_counter() - t0) * 1000,
                    cache_hit=cache_hit,
                    accepted_tokens=len(accepted) + 1,
                )
            )

            if seq.is_finished(None, sampling.max_new_tokens):
                break

            # Overlap: send outcome to draft while next verify could run
            outcome_key = jnp.array([[seq.seq_id, acc, recovery]], dtype=jnp.int32)
            self.draft_worker.submit(
                DraftRequest(
                    command=DraftCommand.SERVE,
                    cache_keys=outcome_key,
                    seq_lens=jnp.array([seq.num_tokens], dtype=jnp.int32),
                    page_tables=page_table[None, :],
                    temperatures=jnp.array([0.0]),
                )
            )
            draft_resp = self.draft_worker.get_response()

        return seq.completion_token_ids

    def shutdown(self) -> None:
        if self.draft_worker:
            self.draft_worker.stop()
