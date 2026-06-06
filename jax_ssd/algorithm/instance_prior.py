"""Instance-SSD: retrieval-based speculation from input context."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np


@dataclass
class ContextSpan:
    start: int
    end: int
    score: float
    tokens: tuple[int, ...]


@dataclass
class InstancePriorConfig:
    span_length: int = 4
    max_candidates: int = 8
    context_ratio: float = 0.3
    min_salience: float = 0.01


def score_token_salience(
    attention_weights: np.ndarray | None,
    context_tokens: np.ndarray,
) -> np.ndarray:
    """Rank input tokens by salience (attention mean or lexical fallback)."""
    n = len(context_tokens)
    if attention_weights is not None and attention_weights.shape[-1] >= n:
        attn = attention_weights[..., :n]
        scores = np.mean(attn, axis=tuple(range(attn.ndim - 1)))
    else:
        # Lexical fallback: identifier-like tokens score higher
        scores = np.ones(n, dtype=np.float32) * 0.1
        for i, tok in enumerate(context_tokens):
            if tok % 7 == 0:  # proxy for identifier tokens in toy vocab
                scores[i] = 1.0
    return scores.astype(np.float32)


def find_high_salience_spans(
    context_tokens: np.ndarray,
    salience: np.ndarray,
    *,
    span_length: int = 4,
    max_spans: int = 8,
) -> list[ContextSpan]:
    """Find top spans to copy-speculate from input context."""
    n = len(context_tokens)
    if n < span_length:
        span = tuple(int(t) for t in context_tokens)
        return [ContextSpan(0, n, float(np.max(salience)), span)]

    spans: list[ContextSpan] = []
    for start in range(n - span_length + 1):
        score = float(np.mean(salience[start : start + span_length]))
        tokens = tuple(int(t) for t in context_tokens[start : start + span_length])
        spans.append(ContextSpan(start, start + span_length, score, tokens))

    spans.sort(key=lambda s: s.score, reverse=True)
    return spans[:max_spans]


def mix_branch_candidates(
    logit_candidates: jnp.ndarray,
    context_candidates: jnp.ndarray,
    *,
    context_ratio: float = 0.3,
) -> jnp.ndarray:
    """Mix Saguaro top-F logit branches with context-derived tokens.

    Args:
        logit_candidates: [B, F] from draft logits
        context_candidates: [B, C] from instance spans

    Returns:
        mixed: [B, F] — quota of context vs logit candidates
    """
    b, f = logit_candidates.shape
    n_context = max(1, int(f * context_ratio))
    n_logit = f - n_context

    logit_part = logit_candidates[:, :n_logit]
    ctx_part = context_candidates[:, :n_context]

    if ctx_part.shape[1] < n_context:
        pad = jnp.zeros((b, n_context - ctx_part.shape[1]), dtype=jnp.int32)
        ctx_part = jnp.concatenate([ctx_part, pad], axis=1)

    return jnp.concatenate([logit_part, ctx_part], axis=1)


def spans_to_speculation_tokens(
    spans: list[ContextSpan],
    k: int,
) -> list[int]:
    """Convert best span to K speculation tokens (pad/truncate)."""
    if not spans:
        return [0] * k
    tokens = list(spans[0].tokens)
    if len(tokens) >= k:
        return tokens[:k]
    return tokens + [tokens[-1]] * (k - len(tokens))
