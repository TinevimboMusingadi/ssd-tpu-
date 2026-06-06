"""Paged KV block allocation."""

from __future__ import annotations

from dataclasses import dataclass, field

from jax_ssd.runtime.sequence import Sequence


@dataclass
class PageManager:
    block_size: int
    num_blocks: int
    free_blocks: list[int] = field(default_factory=list)
    ref_counts: dict[int, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.free_blocks:
            self.free_blocks = list(range(self.num_blocks))

    def allocate(self, seq: Sequence, num_tokens: int) -> bool:
        blocks_needed = (seq.num_tokens + num_tokens + self.block_size - 1) // self.block_size
        current = len(seq.block_table)
        extra = blocks_needed - current
        if extra <= 0:
            return True
        if len(self.free_blocks) < extra:
            return False
        for _ in range(extra):
            block = self.free_blocks.pop()
            seq.block_table.append(block)
            self.ref_counts[block] = self.ref_counts.get(block, 0) + 1
        return True

    def allocate_draft(self, seq: Sequence, num_tokens: int) -> bool:
        blocks_needed = (seq.num_tokens + num_tokens + self.block_size - 1) // self.block_size
        current = len(seq.draft_block_table)
        extra = blocks_needed - current
        if extra <= 0:
            return True
        if len(self.free_blocks) < extra:
            return False
        for _ in range(extra):
            block = self.free_blocks.pop()
            seq.draft_block_table.append(block)
            self.ref_counts[block] = self.ref_counts.get(block, 0) + 1
        return True

    def rollback_speculative(self, seq: Sequence, accepted_tokens: int, reserved: int) -> None:
        """Free blocks reserved for rejected speculative tokens."""
        used = (seq.num_tokens + accepted_tokens + self.block_size - 1) // self.block_size
        total = (seq.num_tokens + reserved + self.block_size - 1) // self.block_size
        while len(seq.block_table) > used and len(seq.block_table) > total - (total - used):
            if len(seq.block_table) > used:
                block = seq.block_table.pop()
                self._free_block(block)

    def free_sequence(self, seq: Sequence) -> None:
        for block in seq.block_table:
            self._free_block(block)
        for block in seq.draft_block_table:
            self._free_block(block)
        seq.block_table.clear()
        seq.draft_block_table.clear()

    def _free_block(self, block: int) -> None:
        self.ref_counts[block] = self.ref_counts.get(block, 1) - 1
        if self.ref_counts[block] <= 0:
            del self.ref_counts[block]
            self.free_blocks.append(block)
