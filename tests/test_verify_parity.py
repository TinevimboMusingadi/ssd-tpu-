"""JAX vs PyTorch verify parity."""

import jax.numpy as jnp
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from jax_ssd.algorithm.verify import verify_greedy
from reference.cuda_ssd.verify_torch import verify_greedy_torch


@pytest.mark.parametrize("batch,k,vocab", [(1, 4, 32), (4, 4, 64), (2, 8, 128)])
def test_greedy_parity(batch, k, vocab):
    rng = np.random.default_rng(42)
    logits_p = rng.standard_normal((batch, k + 1, vocab)).astype(np.float32)
    speculations = rng.integers(0, vocab, (batch, k + 1)).astype(np.int32)

    jax_out = verify_greedy(jnp.array(logits_p), jnp.array(speculations))
    torch_acc, torch_rec = verify_greedy_torch(
        torch.tensor(logits_p), torch.tensor(speculations)
    )

    np.testing.assert_array_equal(np.array(jax_out.accept_until), torch_acc.numpy())
    np.testing.assert_array_equal(np.array(jax_out.recovery_tokens), torch_rec.numpy())
