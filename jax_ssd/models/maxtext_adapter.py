"""MaxText adapter stub — select via SSD_SHARDING_BACKEND=maxtext."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class MaxTextAdapter:
    """Placeholder for MaxText-backed inference (Stage 9)."""

    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            "MaxText adapter is not implemented yet. Set SSD_SHARDING_BACKEND=flax "
            "or install optional maxtext dependencies when available."
        )
