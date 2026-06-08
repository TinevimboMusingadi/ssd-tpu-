"""Patch transformers 5.x for EasyDeL 0.1.4.x (removed working_or_temp_dir)."""

from __future__ import annotations

import contextlib
import tempfile
from pathlib import Path


@contextlib.contextmanager
def _working_or_temp_dir(working_dir: str, use_temp_dir: bool = False):
    if use_temp_dir:
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield tmp_dir
    else:
        Path(working_dir).mkdir(parents=True, exist_ok=True)
        yield working_dir


def _noop_download_url(url: str, *args, **kwargs):
    raise NotImplementedError(f"download_url shim does not fetch: {url}")


def _is_remote_url(url_or_filename: str) -> bool:
    from urllib.parse import urlparse

    parsed = urlparse(str(url_or_filename))
    return parsed.scheme in ("http", "https")


def apply_transformers_compat() -> None:
    import transformers.utils as utils
    import transformers.utils.generic as generic

    if not hasattr(generic, "working_or_temp_dir"):
        generic.working_or_temp_dir = _working_or_temp_dir

    if not hasattr(utils, "download_url"):
        try:
            from huggingface_hub import hf_hub_download as download_url

            utils.download_url = download_url
        except ImportError:
            utils.download_url = _noop_download_url

    if not hasattr(utils, "is_remote_url"):
        utils.is_remote_url = _is_remote_url
