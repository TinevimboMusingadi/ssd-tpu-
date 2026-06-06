"""Request scheduler."""

from __future__ import annotations

from collections import deque

from jax_ssd.config import DecodeMode, SSDConfig
from jax_ssd.runtime.page_manager import PageManager
from jax_ssd.runtime.sequence import Sequence, SequenceStatus


class Scheduler:
    def __init__(self, config: SSDConfig, page_manager: PageManager) -> None:
        self.config = config
        self.page_manager = page_manager
        self.waiting: deque[Sequence] = deque()
        self.running: list[Sequence] = []
        self.finished: list[Sequence] = []
        self._next_id = 0

    def add_request(self, prompt_token_ids: list[int]) -> Sequence:
        seq = Sequence(seq_id=self._next_id, prompt_token_ids=list(prompt_token_ids))
        self._next_id += 1
        self.waiting.append(seq)
        return seq

    def schedule(self) -> list[Sequence]:
        scheduled = []
        while self.waiting and len(self.running) < 16:
            seq = self.waiting.popleft()
            reserve = self._reserve_tokens()
            if not self.page_manager.allocate(seq, reserve):
                self.waiting.appendleft(seq)
                break
            if self.config.mode in (DecodeMode.SD, DecodeMode.SSD, DecodeMode.INSTANCE):
                draft_reserve = reserve + self.config.speculate_k * self.config.mq_len
                if not self.page_manager.allocate_draft(seq, draft_reserve):
                    self.waiting.appendleft(seq)
                    break
            seq.status = SequenceStatus.RUNNING
            self.running.append(seq)
            scheduled.append(seq)
        return scheduled

    def _reserve_tokens(self) -> int:
        k = self.config.speculate_k
        if self.config.mode == DecodeMode.AR:
            return 1
        if self.config.mode == DecodeMode.SD:
            return k + 1
        return k + 1

    def postprocess(self, seq: Sequence, new_tokens: list[int], eos_id: int | None, max_tokens: int) -> None:
        seq.append_tokens(new_tokens)
        if seq.is_finished(eos_id, max_tokens):
            seq.status = SequenceStatus.FINISHED
            self.running.remove(seq)
            self.page_manager.free_sequence(seq)
            self.finished.append(seq)

    def postprocess_speculative(
        self,
        seq: Sequence,
        accepted_tokens: list[int],
        recovery: int,
        reserved: int,
        eos_id: int | None,
        max_tokens: int,
    ) -> None:
        seq.append_tokens(accepted_tokens)
        seq.recovery_token = recovery
        self.page_manager.rollback_speculative(seq, len(accepted_tokens), reserved)
        if seq.is_finished(eos_id, max_tokens):
            seq.status = SequenceStatus.FINISHED
            self.running.remove(seq)
            self.page_manager.free_sequence(seq)
            self.finished.append(seq)
