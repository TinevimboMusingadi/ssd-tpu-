"""PyTorch reference verify for parity tests (minimal port of ssd/utils/verify.py)."""

from __future__ import annotations

import torch


def verify_greedy_torch(
    logits_p: torch.Tensor,
    speculations: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Greedy speculative decoding verification.

    Args:
        logits_p: [B, K+1, V]
        speculations: [B, K+1]

    Returns:
        accept_until: [B] int32
        recovery_tokens: [B] int32
    """
    b, kp1, _ = logits_p.shape
    k = kp1 - 1
    target_tokens = logits_p.argmax(dim=-1)  # [B, K+1]

    draft_tokens = speculations[:, 1 : k + 1]
    target_at_draft = target_tokens[:, :k]
    matches = draft_tokens == target_at_draft

    accept_until = torch.zeros(b, dtype=torch.int32)
    recovery = torch.zeros(b, dtype=torch.int32)
    for i in range(b):
        acc = 0
        for j in range(k):
            if matches[i, j]:
                acc += 1
            else:
                break
        accept_until[i] = acc
        if acc == k:
            recovery[i] = target_tokens[i, k]
        else:
            recovery[i] = target_tokens[i, acc]

    return accept_until, recovery
