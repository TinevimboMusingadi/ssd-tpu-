"""Profiling helpers — gate Pallas and logits compression optimizations."""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class ProfileReport:
    target_verify_ms: list[float] = field(default_factory=list)
    draft_tree_ms: list[float] = field(default_factory=list)
    host_queue_ms: list[float] = field(default_factory=list)

    @property
    def avg_target_verify_ms(self) -> float:
        return sum(self.target_verify_ms) / max(len(self.target_verify_ms), 1)

    @property
    def avg_draft_tree_ms(self) -> float:
        return sum(self.draft_tree_ms) / max(len(self.draft_tree_ms), 1)

    def recommend_optimizations(self) -> list[str]:
        recs = []
        total = self.avg_target_verify_ms + self.avg_draft_tree_ms
        if total > 0 and self.avg_draft_tree_ms / total > 0.4:
            recs.append("Enable Pallas branch attention kernel")
        if self.avg_target_verify_ms / total > 0.25 if total else False:
            recs.append("Consider Pallas paged KV kernel")
        if self.host_queue_ms and sum(self.host_queue_ms) / len(self.host_queue_ms) > 5.0:
            recs.append("Enable greedy logits compression on host queues")
        return recs


class Profiler:
    def __init__(self) -> None:
        self.report = ProfileReport()
        self._t0: float | None = None

    def start(self) -> None:
        self._t0 = time.perf_counter()

    def stop_target_verify(self) -> None:
        if self._t0 is not None:
            self.report.target_verify_ms.append((time.perf_counter() - self._t0) * 1000)

    def stop_draft_tree(self) -> None:
        if self._t0 is not None:
            self.report.draft_tree_ms.append((time.perf_counter() - self._t0) * 1000)
