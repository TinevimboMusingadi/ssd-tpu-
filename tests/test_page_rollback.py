from jax_ssd.runtime.page_manager import PageManager
from jax_ssd.runtime.sequence import Sequence


def test_allocate_and_rollback():
    pm = PageManager(block_size=16, num_blocks=10)
    seq = Sequence(seq_id=0, prompt_token_ids=list(range(20)))
    assert pm.allocate(seq, 32)
    n_after = len(seq.block_table)
    pm.rollback_speculative(seq, accepted_tokens=4, reserved=20)
    assert len(seq.block_table) <= n_after
