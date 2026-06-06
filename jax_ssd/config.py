"""Runtime configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence

from connect.mesh_allocator import MeshAllocation


def _env_bool(key: str, default: bool = False) -> bool:
    return os.getenv(key, "1" if default else "0").lower() in ("1", "true", "yes")


def _default_target_path() -> str | None:
    return os.getenv("TARGET_MODEL_PATH", "./models/google_gemma-2b-it")


def _default_draft_path() -> str | None:
    return os.getenv("DRAFT_MODEL_PATH") or _default_target_path()


class DecodeMode(str, Enum):
    AR = "ar"
    SD = "sd"
    SSD = "ssd"
    INSTANCE = "instance"


@dataclass
class SSDConfig:
    mode: DecodeMode = DecodeMode.AR
    speculate_k: int = 4
    fan_out_list: tuple[int, ...] = (2, 2, 2, 2, 2)
    sampler_x: float = 0.5
    max_model_len: int = 4096
    block_size: int = 16
    num_blocks: int = 256
    batch_buckets: tuple[int, ...] = (1, 2, 4, 8, 16)
    target_model_path: str | None = field(default_factory=_default_target_path)
    draft_model_path: str | None = field(default_factory=_default_draft_path)
    use_toy_model: bool = field(default_factory=lambda: _env_bool("SSD_USE_TOY_MODEL"))
    mesh_allocation: MeshAllocation | None = None
    instance_context_ratio: float = 0.3

    @property
    def mq_len(self) -> int:
        k = self.speculate_k
        fan = self.fan_out_list
        if len(fan) < k + 1:
            fan = fan + (fan[-1],) * (k + 1 - len(fan))
        return sum(fan[: k + 1])

    @classmethod
    def from_fanout_string(cls, fanout: str, **kwargs) -> SSDConfig:
        fl = tuple(int(x.strip()) for x in fanout.split(",") if x.strip())
        return cls(fan_out_list=fl, **kwargs)
