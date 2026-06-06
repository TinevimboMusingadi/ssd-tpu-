"""Generation controls."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SamplingParams:
    temperature: float = 0.0
    draft_temperature: float = 0.0
    max_new_tokens: int = 128
    ignore_eos: bool = False
