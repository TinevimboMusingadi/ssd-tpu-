"""Inference metrics collection."""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class StepMetrics:
    target_verify_ms: float = 0.0
    draft_step_ms: float = 0.0
    cache_hit: bool = False
    accepted_tokens: int = 0


@dataclass
class RunMetrics:
    mode: str = "ar"
    total_tokens: int = 0
    total_time_s: float = 0.0
    ttft_s: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    steps: list[StepMetrics] = field(default_factory=list)

    @property
    def tokens_per_second(self) -> float:
        if self.total_time_s <= 0:
            return 0.0
        return self.total_tokens / self.total_time_s

    @property
    def cache_hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        if total == 0:
            return 0.0
        return self.cache_hits / total

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "total_tokens": self.total_tokens,
            "total_time_s": self.total_time_s,
            "tokens_per_second": self.tokens_per_second,
            "ttft_s": self.ttft_s,
            "cache_hit_rate": self.cache_hit_rate,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "num_steps": len(self.steps),
        }


class MetricsCollector:
    def __init__(self, mode: str) -> None:
        self.metrics = RunMetrics(mode=mode)
        self._start: float | None = None
        self._first_token: float | None = None

    def start(self) -> None:
        self._start = time.perf_counter()

    def record_first_token(self) -> None:
        if self._first_token is None and self._start is not None:
            self._first_token = time.perf_counter()
            self.metrics.ttft_s = self._first_token - self._start

    def record_step(self, step: StepMetrics) -> None:
        self.metrics.steps.append(step)
        self.metrics.total_tokens += step.accepted_tokens
        if step.cache_hit:
            self.metrics.cache_hits += 1
        else:
            self.metrics.cache_misses += 1

    def finish(self) -> RunMetrics:
        if self._start is not None:
            self.metrics.total_time_s = time.perf_counter() - self._start
        return self.metrics
