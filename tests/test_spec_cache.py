import jax.numpy as jnp

from jax_ssd.algorithm.spec_cache import SpecCache


def test_cache_hit_and_miss():
    cache = SpecCache.create(num_branches=8, k=4, vocab=32)
    key = jnp.array([1, 2, 10], dtype=jnp.int32)
    tokens = jnp.array([5, 6, 7, 8], dtype=jnp.int32)
    logits = jnp.zeros((4, 32), dtype=jnp.float32)
    cache = cache.insert(0, key, tokens, logits)

    hit, idx, out_tok, _ = cache.lookup(key[None, :])
    assert bool(hit[0])
    assert int(idx[0]) == 0
    assert list(map(int, out_tok[0])) == [5, 6, 7, 8]

    miss_key = jnp.array([[9, 9, 9]], dtype=jnp.int32)
    hit2, _, _, _ = cache.lookup(miss_key)
    assert not bool(hit2[0])
