"""JAX port of speculative decoding acceptance rules."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class VerifyResult:
    accept_until: jax.Array  # [B] int32 — number of draft tokens accepted (0..K)
    recovery_tokens: jax.Array  # [B] int32
    accepted_mask: jax.Array  # [B, K+1] bool


def verify_greedy(
    logits_p: jax.Array,
    speculations: jax.Array,
    *,
    cache_hit: jax.Array | None = None,
) -> VerifyResult:
    """Greedy verification over K+1 positions.

    Args:
        logits_p: [B, K+1, V] target logits
        speculations: [B, K+1] — index 0 is prior recovery; 1..K are draft tokens
    """
    b, kp1, _ = logits_p.shape
    k = kp1 - 1
    target_tokens = jnp.argmax(logits_p, axis=-1).astype(jnp.int32)

    draft_tokens = speculations[:, 1:]
    target_at_draft = target_tokens[:, :k]
    matches = draft_tokens == target_at_draft

    cum = jnp.cumprod(matches.astype(jnp.float32), axis=1)
    accept_until = jnp.sum(cum, axis=1).astype(jnp.int32)

    all_accepted = accept_until == k
    recovery_if_all = target_tokens[:, k]
    recovery_if_partial = jnp.take_along_axis(
        target_tokens, accept_until[:, None], axis=1
    ).squeeze(1)
    recovery_tokens = jnp.where(all_accepted, recovery_if_all, recovery_if_partial)

    positions = jnp.arange(k)[None, :]
    accepted_mask = jnp.zeros((b, kp1), dtype=bool)
    accepted_mask = accepted_mask.at[:, 1:].set(positions < accept_until[:, None])

    if cache_hit is not None:
        pass  # cache_hit affects sampling path; greedy unchanged

    return VerifyResult(
        accept_until=accept_until,
        recovery_tokens=recovery_tokens.astype(jnp.int32),
        accepted_mask=accepted_mask,
    )


def _softmax(logits: jax.Array) -> jax.Array:
    logits = logits - jnp.max(logits, axis=-1, keepdims=True)
    exp = jnp.exp(logits)
    return exp / jnp.sum(exp, axis=-1, keepdims=True)


def verify_sample(
    logits_p: jax.Array,
    logits_q: jax.Array,
    speculations: jax.Array,
    key: jax.Array,
    *,
    temperature_p: float = 1.0,
    temperature_q: float = 1.0,
    sampler_x: float = 0.5,
) -> VerifyResult:
    """Sampling-mode verification (row-wise for correctness)."""
    b = logits_p.shape[0]
    keys = jax.random.split(key, b)
    results = [
        _verify_sample_row(logits_p[i], logits_q[i], speculations[i], keys[i], temperature_p, temperature_q, sampler_x)
        for i in range(b)
    ]
    accept_until = jnp.array([r[0] for r in results], dtype=jnp.int32)
    recovery_tokens = jnp.array([r[1] for r in results], dtype=jnp.int32)
    k = logits_p.shape[1] - 1
    positions = jnp.arange(k)[None, :]
    accepted_mask = jnp.zeros((b, k + 1), dtype=bool)
    accepted_mask = accepted_mask.at[:, 1:].set(positions < accept_until[:, None])
    return VerifyResult(
        accept_until=accept_until,
        recovery_tokens=recovery_tokens,
        accepted_mask=accepted_mask,
    )


def _verify_sample_row(lp, lq, spec, key, temp_p, temp_q, sampler_x):
    k = lp.shape[0] - 1
    keys = jax.random.split(key, k + 2)
    accept = 0
    for j in range(k):
        p = _softmax(lp[j + 1] / temp_p)
        q = _softmax(lq[j] / temp_q)
        tok = int(spec[j + 1])
        ratio = jnp.minimum(1.0, p[tok] / jnp.maximum(q[tok], 1e-12))
        if float(jax.random.uniform(keys[j])) < float(ratio):
            accept += 1
        else:
            break
    if accept == k:
        p_final = _softmax(lp[k] / temp_p)
        recovery = int(jax.random.categorical(keys[k + 1], jnp.log(p_final + 1e-12)))
    else:
        p_rej = _softmax(lp[accept + 1] / temp_p)
        q_rej = _softmax(lq[accept] / temp_q)
        residual = jnp.maximum(p_rej - sampler_x * q_rej, 0.0)
        residual = residual / jnp.maximum(jnp.sum(residual), 1e-12)
        recovery = int(jax.random.categorical(keys[k + 1], jnp.log(residual + 1e-12)))
    return accept, recovery
