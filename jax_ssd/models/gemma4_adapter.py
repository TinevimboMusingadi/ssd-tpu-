"""Gemma 4 on TPU via EasyDeL (text-only AR path)."""

from __future__ import annotations

import logging
import os
from typing import Callable

import jax
import jax.numpy as jnp

from connect.gcs_storage import resolve_model_path
from jax_ssd.kernels.kv_cache import KVCacheState
from jax_ssd.models.base import DecodeResult, PrefillResult, VerifyResultModel

logger = logging.getLogger(__name__)


def _load_easydel_model(ed, ed_repo: str, n_devices: int):
    """Load sharded Gemma 4 weights (EasyDeL 0.1.4.x API)."""
    fsdp = max(1, min(n_devices, 4))
    dtype = jnp.bfloat16
    shard4 = (1, fsdp, 1, 1)
    shard5 = (1, 1, 1, fsdp, 1)
    for shard, names in (
        (shard5, ("dp", "fsdp", "ep", "tp", "sp")),
        (shard4, ("dp", "fsdp", "tp", "sp")),
    ):
        kwargs = dict(
            dtype=dtype,
            param_dtype=dtype,
            sharding_axis_dims=shard,
            sharding_axis_names=names,
            auto_shard_model=True,
        )
        for cls_name in ("AutoEasyDeLModelForImageTextToText", "AutoEasyDeLModelForCausalLM"):
            cls = getattr(ed, cls_name, None)
            if cls is None:
                continue
            try:
                return cls.from_pretrained(ed_repo, **kwargs)
            except Exception as exc:
                logger.warning("EasyDeL %s load failed for %s: %s", cls_name, ed_repo, exc)
    raise RuntimeError(f"Could not load EasyDeL model {ed_repo}")


def _build_gemma4_engine(ed, ed_repo: str, n_devices: int, tokenizer=None):
    """Create EasyDeL inference engine (new eSurge API or legacy vInference)."""
    max_len = int(os.getenv("GEMMA4_MAX_MODEL_LEN", "2048"))
    fsdp = max(1, min(n_devices, 4))
    shard5 = (1, 1, 1, fsdp, 1)

    model = _load_easydel_model(ed, ed_repo, n_devices)

    if hasattr(ed, "eSurge"):
        try:
            return ed.eSurge(
                model=model,
                processor=tokenizer,
                max_model_len=max_len,
                max_num_seqs=4,
                hbm_utilization=float(os.getenv("GEMMA4_HBM_UTIL", "0.85")),
                sharding_axis_dims=shard5,
            )
        except TypeError:
            pass
    return ed.vInference(model=model, max_new_tokens=max_len)


def _load_gemma4_tokenizer(tok_repo: str, hf_token: str | None):
    """Text-only tokenizer without Gemma4Processor (avoids torchvision/PIL)."""
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(tok_repo, token=hf_token)
    except Exception:
        from huggingface_hub import hf_hub_download

        sp = hf_hub_download(tok_repo, "tokenizer.model", token=hf_token)
        return AutoTokenizer.from_pretrained(tok_repo, tokenizer_file=sp, token=hf_token)


def _resolve_hf_or_local(model_path: str) -> str:
    """HF hub id (org/name) or local/GCS path."""
    from pathlib import Path

    if model_path.startswith("gs://"):
        return resolve_model_path(model_path)
    if Path(model_path).exists():
        return model_path
    if "/" in model_path and not model_path.startswith((".", "/", "~")):
        return model_path
    return resolve_model_path(model_path)

# HF ids (Apache 2.0, not gated) -> EasyDeL JAX checkpoints
_GEMMA4_EASYDEL: dict[str, str] = {
    "google/gemma-4-E2B-it": "EasyDeL/gemma-4-E2B-it",
    "google/gemma-4-E4B-it": "EasyDeL/gemma-4-E4B-it",
    "google/gemma-4-E2B-it-assistant": "EasyDeL/gemma-4-E2B-it-assistant",
    "google/gemma-4-E4B-it-assistant": "EasyDeL/gemma-4-E4B-it-assistant",
}

_SHARED: dict[str, object] = {}


def _easydel_repo(model_path: str) -> str:
    path = model_path.rstrip("/")
    name = path.split("/")[-1] if "/" in path else path
    for key, repo in _GEMMA4_EASYDEL.items():
        if key.endswith(name) or name in key.replace("/", "_"):
            return repo
    if path.startswith("EasyDeL/"):
        return path
    if "E4B" in name or "e4b" in name.lower():
        return "EasyDeL/gemma-4-E4B-it"
    if "assistant" in name.lower():
        return "EasyDeL/gemma-4-E2B-it-assistant"
    return "EasyDeL/gemma-4-E2B-it"


def _tokenizer_repo(model_path: str) -> str:
    repo = _easydel_repo(model_path)
    if "assistant" in repo:
        return "google/gemma-4-E2B-it-assistant"
    if "E4B" in repo:
        return "google/gemma-4-E4B-it"
    return "google/gemma-4-E2B-it"


class Gemma4Adapter:
    """Gemma 4 via EasyDeL eSurge — AR generation; SD/SSD uses toy draft until wired."""

    def __init__(
        self,
        model_path: str,
        vocab_size: int = 262_144,
        share_key: str | None = None,
        mesh=None,
        devices: tuple[jax.Device, ...] | None = None,
        role: str = "target",
    ) -> None:
        from jax_ssd.compat.transformers_shim import apply_transformers_compat

        apply_transformers_compat()
        os.environ.setdefault("EASYDEL_AUTO", "1")
        os.environ.setdefault("ENABLE_DISTRIBUTED_INIT", "0")
        import easydel as ed
        if hasattr(ed, "SamplingParams"):
            self._SamplingParams = ed.SamplingParams
        else:
            from easydel.inference.sampling_params import SamplingParams as _SP

            self._SamplingParams = _SP
        local_path = _resolve_hf_or_local(model_path)
        self.model_path = model_path
        self.local_path = local_path
        self._role = role
        self._devices = devices or tuple(jax.devices())
        n = max(1, len(self._devices))

        ed_repo = _easydel_repo(model_path)
        tok_repo = _tokenizer_repo(model_path)
        key = share_key or ed_repo

        hf_token = (os.getenv("HF_TOKEN") or "").strip() or None
        self._tokenizer = _load_gemma4_tokenizer(tok_repo, hf_token)
        self.vocab_size = int(getattr(self._tokenizer, "vocab_size", vocab_size) or vocab_size)

        if key in _SHARED:
            self._esurge = _SHARED[key]
            logger.info("Reusing Gemma4 engine for %s (%s)", ed_repo, role)
        else:
            logger.info("Loading Gemma4 %s on %d devices (%s)...", ed_repo, n, role)
            self._esurge = _build_gemma4_engine(ed, ed_repo, n, self._tokenizer)
            _SHARED[key] = self._esurge

        self._dummy_kv = KVCacheState.allocate(
            num_layers=1, num_blocks=1, block_size=1, num_kv_heads=1, head_dim=1
        )

    def reset_cache(self) -> None:
        pass

    def allocate_kv(self) -> KVCacheState:
        return self._dummy_kv

    def tokenize(self, text: str) -> list[int]:
        ids = self._tokenizer.encode(text, add_special_tokens=True)
        return list(ids)

    def decode_tokens_to_str(self, token_ids: list[int]) -> str:
        return self._tokenizer.decode(token_ids, skip_special_tokens=True)

    def _generated_token_ids(self, prompt_tokens: list[int], sequences) -> list[int]:
        seq = jnp.asarray(sequences)[0]
        prompt_len = len(prompt_tokens)
        return [int(t) for t in seq[prompt_len:]]

    def generate_ar(
        self,
        prompt_tokens: list[int],
        max_new_tokens: int,
        on_token: Callable[[int, str], None] | None = None,
    ) -> list[int]:
        """Full AR generation via EasyDeL (vInference or eSurge)."""
        sampling = self._SamplingParams(max_tokens=max_new_tokens, temperature=0.0, top_p=1.0)
        engine = self._esurge

        if hasattr(engine, "generate") and not hasattr(engine, "model"):
            prompt = self._tokenizer.decode(prompt_tokens, skip_special_tokens=False)
            outputs = engine.generate(prompt, sampling_params=sampling)
            out = outputs[0].outputs[0]
            new_text = getattr(out, "text", str(out))
            new_ids = self._tokenizer.encode(new_text, add_special_tokens=False)
        else:
            input_ids = jnp.array([prompt_tokens], dtype=jnp.int32)
            final = None
            for state in engine.generate(input_ids, sampling_params=sampling):
                final = state
            if final is None:
                return []
            sequences = getattr(final, "sequences", None) or getattr(final, "output_ids", None)
            if sequences is None:
                raise RuntimeError("EasyDeL generate returned no sequences")
            new_ids = self._generated_token_ids(prompt_tokens, sequences)

        generated: list[int] = []
        for tid in new_ids:
            generated.append(int(tid))
            if on_token:
                on_token(int(tid), self.decode_tokens_to_str([int(tid)]))
        return generated

    # Protocol stubs — AR uses generate_ar(); SD/SSD not yet on Gemma4 forward
    def prefill(self, token_ids, kv_cache, page_table) -> PrefillResult:
        logits = jnp.zeros((len(token_ids), self.vocab_size), dtype=jnp.float32)
        return PrefillResult(logits=logits, kv_cache=kv_cache)

    def decode(self, token_id, position, kv_cache, page_table) -> DecodeResult:
        logits = jnp.zeros((1, self.vocab_size), dtype=jnp.float32)
        return DecodeResult(logits=logits, kv_cache=kv_cache, next_token=int(token_id))

    def verify(self, token_ids_kp1, positions, kv_cache, page_table) -> VerifyResultModel:
        logits = jnp.zeros((1, len(token_ids_kp1), self.vocab_size), dtype=jnp.float32)
        return VerifyResultModel(logits=logits, kv_cache=kv_cache)
