# Algorithms

## AR
Target-only autoregressive decode.

## SD
Draft proposes K tokens; target verifies in one pass. Sequential.

## SSD / Saguaro
Draft and target overlap. Draft pre-speculates for predicted verification outcomes keyed by `[seq_id, accept_len, recovery_token]`.

## Instance-SSD
For code refactoring: copy-speculate high-salience spans from input context instead of draft forward passes where possible. Mixed with Saguaro logit branches (70/30 default).
