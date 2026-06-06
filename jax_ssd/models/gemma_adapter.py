"""Gemma model adapter — Flax causal LM on TPU with optional mesh sharding."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax.sharding import Mesh

from connect.gcs_storage import resolve_model_path
from jax_ssd.kernels.kv_cache import KVCacheState
from jax_ssd.models.base import DecodeResult, PrefillResult, VerifyResultModel
from jax_ssd.models.sharding import fsdp_shard_params, primary_device

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

_SHARED: dict[str, tuple[object, object]] = {}


class GemmaAdapter:
    """Real Gemma inference via HuggingFace Flax weights."""

    def __init__(
        self,
        model_path: str,
        vocab_size: int = 256_000,
        share_key: str | None = None,
        mesh: Mesh | None = None,
        devices: tuple[jax.Device, ...] | None = None,
        role: str = "target",
    ) -> None:
        from transformers import AutoTokenizer, FlaxAutoModelForCausalLM

        local_path = resolve_model_path(model_path)
        self.model_path = model_path
        self.local_path = local_path
        self._mesh = mesh
        self._devices = devices or ()
        self._role = role
        self._device = primary_device(self._devices) if self._devices else None

        self._tokenizer = AutoTokenizer.from_pretrained(local_path)
        key = share_key or model_path
        if key in _SHARED:
            self.model, self.params = _SHARED[key]
            logger.info("Reusing loaded Gemma weights for %s (%s)", model_path, role)
        else:
            logger.info(
                "Loading Gemma %s from %s (mesh=%s, devices=%d)...",
                role,
                local_path,
                mesh.axis_names if mesh else None,
                len(self._devices),
            )
            with jax.default_device(self._device) if self._device else _nullcontext():
                self.model = FlaxAutoModelForCausalLM.from_pretrained(
                    local_path,
                    dtype=jnp.bfloat16,
                )
                self.params = fsdp_shard_params(self.model.params, mesh)
            _SHARED[key] = (self.model, self.params)

        self.vocab_size = int(getattr(self.model.config, "vocab_size", vocab_size))
        self._past_key_values: tuple | None = None
        self._dummy_kv = KVCacheState.allocate(
            num_layers=1, num_blocks=1, block_size=1, num_kv_heads=1, head_dim=1
        )

    def reset_cache(self) -> None:
        self._past_key_values = None

    def allocate_kv(self) -> KVCacheState:
        self.reset_cache()
        return self._dummy_kv

    def _run_forward(
        self,
        input_ids: jax.Array,
        *,
        past_key_values: tuple | None = None,
        update_cache: bool = False,
    ) -> jax.Array:
        def _call():
            outputs = self.model(
                input_ids=input_ids,
                past_key_values=past_key_values,
                params=self.params,
                train=False,
            )
            if update_cache:
                self._past_key_values = outputs.past_key_values
            return outputs.logits.astype(jnp.float32)

        if self._mesh is not None:
            with jax.set_mesh(self._mesh):
                return _call()
        return _call()

    def prefill(
        self,
        token_ids: jnp.ndarray,
        kv_cache: KVCacheState,
        page_table: jnp.ndarray,
    ) -> PrefillResult:
        input_ids = jnp.asarray(token_ids, dtype=jnp.int32)[None, :]
        logits = self._run_forward(input_ids, update_cache=True)
        return PrefillResult(logits=logits[0], kv_cache=kv_cache)

    def decode(
        self,
        token_id: int,
        position: int,
        kv_cache: KVCacheState,
        page_table: jnp.ndarray,
    ) -> DecodeResult:
        input_ids = jnp.array([[int(token_id)]], dtype=jnp.int32)
        logits = self._run_forward(
            input_ids,
            past_key_values=self._past_key_values,
            update_cache=True,
        )
        next_token = int(jnp.argmax(logits[0, -1]))
        return DecodeResult(logits=logits[0], kv_cache=kv_cache, next_token=next_token)

    def verify(
        self,
        token_ids_kp1: jnp.ndarray,
        positions: jnp.ndarray,
        kv_cache: KVCacheState,
        page_table: jnp.ndarray,
    ) -> VerifyResultModel:
        input_ids = jnp.asarray(token_ids_kp1, dtype=jnp.int32)[None, :]
        logits = self._run_forward(
            input_ids,
            past_key_values=self._past_key_values,
            update_cache=False,
        )
        return VerifyResultModel(logits=logits, kv_cache=kv_cache)

    def commit_tokens(self, token_ids: list[int]) -> None:
        for tok in token_ids:
            input_ids = jnp.array([[int(tok)]], dtype=jnp.int32)
            self._run_forward(
                input_ids,
                past_key_values=self._past_key_values,
                update_cache=True,
            )

    def draft_speculate(
        self,
        prefix_tokens: list[int],
        k: int,
    ) -> tuple[list[int], jnp.ndarray]:
        saved = self._past_key_values
        self.reset_cache()
        pre = self.prefill(
            jnp.array(prefix_tokens, dtype=jnp.int32),
            self._dummy_kv,
            jnp.zeros((1,), dtype=jnp.int32),
        )
        token = int(jnp.argmax(pre.logits[-1]))
        logits_list = []
        spec: list[int] = []
        for _ in range(k):
            dec = self.decode(token, 0, self._dummy_kv, jnp.zeros((1,), dtype=jnp.int32))
            logits_list.append(dec.logits[-1])
            token = dec.next_token
            spec.append(token)
        self._past_key_values = saved
        return spec, jnp.stack(logits_list)

    def decode_tokens_to_str(self, token_ids: list[int]) -> str:
        return self._tokenizer.decode(token_ids, skip_special_tokens=True)

    def tokenize(self, text: str) -> list[int]:
        return list(self._tokenizer.encode(text, add_special_tokens=True))


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *args):
        return False
