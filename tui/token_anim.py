"""Live token stream state for TUI panels."""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class TokenStream:
    """Accumulates decoded text and metrics as tokens arrive."""

    mode: str
    text: str = ""
    token_count: int = 0
    tokens_per_second: float = 0.0
    cache_hit_rate: float = 0.0
    cache_hits: int = 0
    cache_rounds: int = 0
    _start: float = field(default_factory=time.perf_counter)
    _last_flash: float = 0.0

    def append(self, decoded: str) -> None:
        self.text += decoded
        self.token_count += 1
        elapsed = time.perf_counter() - self._start
        if elapsed > 0:
            self.tokens_per_second = self.token_count / elapsed
        self._last_flash = time.perf_counter()

    def record_cache_hit(self, hit: bool) -> None:
        self.cache_rounds += 1
        if hit:
            self.cache_hits += 1
        if self.cache_rounds:
            self.cache_hit_rate = self.cache_hits / self.cache_rounds

    @property
    def display_text(self) -> str:
        return self.text + "▌"

    def metrics_line(self) -> str:
        extra = ""
        if self.mode in ("ssd", "instance"):
            label = "hit" if self.mode == "ssd" else "span"
            extra = f" · {label} {self.cache_hit_rate:.0%}"
        return f"{self.tokens_per_second:.0f} t/s · {self.token_count}{extra}"

    def should_flash(self) -> bool:
        return (time.perf_counter() - self._last_flash) < 0.15
