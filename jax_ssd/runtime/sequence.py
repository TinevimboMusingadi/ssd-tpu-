"""Per-request sequence state."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class SequenceStatus(str, Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"
    PREEMPTED = "preempted"


@dataclass
class Sequence:
    seq_id: int
    prompt_token_ids: list[int]
    status: SequenceStatus = SequenceStatus.WAITING
    completion_token_ids: list[int] = field(default_factory=list)
    block_table: list[int] = field(default_factory=list)
    draft_block_table: list[int] = field(default_factory=list)
    recovery_token: int = 0
    num_cached_tokens: int = 0

    @property
    def token_ids(self) -> list[int]:
        return self.prompt_token_ids + self.completion_token_ids

    @property
    def num_tokens(self) -> int:
        return len(self.token_ids)

    @property
    def num_completion_tokens(self) -> int:
        return len(self.completion_token_ids)

    def append_token(self, token: int) -> None:
        self.completion_token_ids.append(token)

    def append_tokens(self, tokens: list[int]) -> None:
        self.completion_token_ids.extend(tokens)

    def is_finished(self, eos_id: int | None, max_tokens: int) -> bool:
        if self.num_completion_tokens >= max_tokens:
            return True
        if eos_id is not None and self.completion_token_ids and self.completion_token_ids[-1] == eos_id:
            return True
        return False
