"""Public LLM API."""

from __future__ import annotations

from collections.abc import Callable
from typing import Iterator

from jax_ssd.config import DecodeMode, SSDConfig
from jax_ssd.runtime.engine import LLMEngine
from jax_ssd.runtime.metrics import RunMetrics
from jax_ssd.sampling_params import SamplingParams

TokenCallback = Callable[[int, str], None]


class LLM:
    def __init__(self, config: SSDConfig | None = None, **kwargs) -> None:
        if config is None:
            config = SSDConfig(**kwargs)
        self.config = config
        self._engine = LLMEngine(config)

    @classmethod
    def from_mode(cls, mode: str | DecodeMode, **kwargs) -> LLM:
        if isinstance(mode, str):
            mode = DecodeMode(mode)
        return cls(SSDConfig(mode=mode, **kwargs))

    def generate(
        self,
        prompts: list[list[int]] | list[str],
        sampling_params: SamplingParams | None = None,
        *,
        on_token: TokenCallback | None = None,
    ) -> tuple[list[list[int]], RunMetrics]:
        sampling = sampling_params or SamplingParams()
        if prompts and isinstance(prompts[0], str):
            prompts = [self._tokenize(p) for p in prompts]  # type: ignore[arg-type]
        return self._engine.generate(prompts, sampling, on_token=on_token)

    def generate_stream(
        self,
        prompt: list[int] | str,
        sampling_params: SamplingParams | None = None,
        on_token: TokenCallback | None = None,
    ) -> Iterator[tuple[int, str]]:
        sampling = sampling_params or SamplingParams()
        if isinstance(prompt, str):
            prompt = self._tokenize(prompt)
        return self._engine.generate_stream(prompt, sampling, on_token=on_token)

    def _tokenize(self, text: str) -> list[int]:
        tokenize = getattr(self._engine.target, "tokenize", None)
        if tokenize is not None:
            return tokenize(text)
        return [ord(c) % 64 for c in text[:128]]

    def shutdown(self) -> None:
        self._engine.shutdown()
