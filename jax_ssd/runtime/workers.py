"""Target and draft workers with host queues for async SSD."""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass

import jax.numpy as jnp

from jax_ssd.algorithm.async_protocol import DraftCommand, DraftRequest, DraftResponse, VerifyOutcome
from jax_ssd.algorithm.branch_prior import top_f_recovery_tokens
from jax_ssd.algorithm.instance_prior import (
    find_high_salience_spans,
    score_token_salience,
    spans_to_speculation_tokens,
)
from jax_ssd.algorithm.spec_cache import SpecCache
from jax_ssd.algorithm.verify import verify_greedy
from jax_ssd.config import DecodeMode, SSDConfig
from jax_ssd.models.base import DecodeModelAdapter
from jax_ssd.runtime.metrics import StepMetrics


@dataclass
class WorkerConfig:
    use_toy: bool = True
    instance_mode: bool = False


class DraftWorker:
    """Draft speculator: serves cache hits, rebuilds tree cache."""

    def __init__(self, config: SSDConfig, draft_model: DecodeModelAdapter | None = None) -> None:
        self.config = config
        from jax_ssd.models.toy_model import ToyModelAdapter

        self.model = draft_model or ToyModelAdapter()
        self.request_queue: queue.Queue = queue.Queue()
        self.response_queue: queue.Queue = queue.Queue()
        self.cache: SpecCache | None = None
        self._thread: threading.Thread | None = None
        self._running = False
        self._context_tokens: list[int] = []

    def start(self) -> None:
        self._running = True
        self.cache = SpecCache.create(
            num_branches=self.config.mq_len * 4,
            k=self.config.speculate_k,
            vocab=self.model.vocab_size,
        )
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)

    def set_context(self, tokens: list[int]) -> None:
        self._context_tokens = list(tokens)

    def submit(self, req: DraftRequest) -> None:
        self.request_queue.put(req)

    def get_response(self, timeout: float = 30.0) -> DraftResponse:
        return self.response_queue.get(timeout=timeout)

    def _loop(self) -> None:
        while self._running:
            try:
                req = self.request_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if req.command == DraftCommand.SHUTDOWN:
                break
            t0 = time.perf_counter()
            resp = self._handle(req)
            resp  # noqa: B018
            self.response_queue.put(resp)
            _ = time.perf_counter() - t0

    def _handle(self, req: DraftRequest) -> DraftResponse:
        assert self.cache is not None
        k = self.config.speculate_k
        b = req.cache_keys.shape[0]

        hit, _, tokens, logits = self.cache.lookup(req.cache_keys)

        if not bool(jnp.any(hit)):
            # JIT fallback: run quick draft speculation
            spec_tokens = []
            spec_logits = []
            for i in range(b):
                prefix = [int(req.cache_keys[i, 2])]
                if self.config.mode == DecodeMode.INSTANCE and self._context_tokens:
                    sal = score_token_salience(None, jnp.array(self._context_tokens))
                    spans = find_high_salience_spans(
                        jnp.array(self._context_tokens), sal, span_length=k
                    )
                    spec = spans_to_speculation_tokens(spans, k)
                else:
                    spec, _ = self.model.draft_speculate(prefix, k)
                spec_tokens.append(spec)
                spec_logits.append(
                    jnp.zeros((k, self.model.vocab_size), dtype=jnp.float32)
                )
            tokens = jnp.array(spec_tokens, dtype=jnp.int32)
            logits = jnp.stack(spec_logits)
            hit = jnp.zeros((b,), dtype=bool)

        # Rebuild cache for next round (while target would verify)
        self.cache = self.cache.reset()
        self._rebuild_cache(req)

        return DraftResponse(cache_hit=hit, spec_tokens=tokens, draft_logits=logits)

    def _rebuild_cache(self, req: DraftRequest) -> None:
        assert self.cache is not None
        k = self.config.speculate_k
        fan_list = self.config.fan_out_list
        slot = 0
        for acc_idx, fan in enumerate(fan_list[: k + 1]):
            glue_logits = jnp.zeros((self.model.vocab_size,))
            candidates = top_f_recovery_tokens(glue_logits[None, :], fan)
            for f in range(fan):
                key = req.cache_keys[0].at[1].set(acc_idx).at[2].set(candidates[0, f])
                prefix = [int(key[2])]
                if self.config.mode == DecodeMode.INSTANCE and self._context_tokens:
                    sal = score_token_salience(None, jnp.array(self._context_tokens))
                    spans = find_high_salience_spans(
                        jnp.array(self._context_tokens), sal, span_length=k
                    )
                    spec = spans_to_speculation_tokens(spans, k)
                else:
                    spec, _ = self.model.draft_speculate(prefix, k)
                self.cache = self.cache.insert(
                    slot,
                    key,
                    jnp.array(spec, dtype=jnp.int32),
                    jnp.zeros((k, self.model.vocab_size)),
                )
                slot += 1


# Fix typo in DraftWorker.start - used `config` instead of `self.config`