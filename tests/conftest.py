"""Pytest defaults: fast toy backend unless real weights are explicitly requested."""

from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True)
def _use_toy_model(monkeypatch: pytest.MonkeyPatch) -> None:
    if os.getenv("SSD_USE_REAL_MODEL") not in ("1", "true", "yes"):
        monkeypatch.setenv("SSD_USE_TOY_MODEL", "1")
