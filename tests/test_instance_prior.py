import numpy as np

from jax_ssd.algorithm.instance_prior import (
    find_high_salience_spans,
    score_token_salience,
    spans_to_speculation_tokens,
)


def test_instance_span_selection():
    tokens = np.arange(20, dtype=np.int32)
    sal = score_token_salience(None, tokens)
    spans = find_high_salience_spans(tokens, sal, span_length=4, max_spans=3)
    assert len(spans) >= 1
    spec = spans_to_speculation_tokens(spans, k=4)
    assert len(spec) == 4
