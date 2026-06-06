"""Model + slice profiles for SSD-TPU."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelProfile:
    name: str
    target_repo: str
    draft_repo: str
    min_chips: int
    description: str

    def target_gcs_path(self, bucket: str, prefix: str = "models") -> str:
        bucket = bucket.rstrip("/")
        return f"{bucket}/{prefix}/{self.target_repo.replace('/', '_')}"

    def draft_gcs_path(self, bucket: str, prefix: str = "models") -> str:
        bucket = bucket.rstrip("/")
        return f"{bucket}/{prefix}/{self.draft_repo.replace('/', '_')}"


PROFILES: dict[str, ModelProfile] = {
    "sd-pair-7b": ModelProfile(
        name="sd-pair-7b",
        target_repo="google/gemma-7b-it",
        draft_repo="google/gemma-2b-it",
        min_chips=8,
        description="Gemma-7B + Gemma-2B on one v6e-8 VM (7+1 chip split)",
    ),
    "sd-pair-7b-dual8": ModelProfile(
        name="sd-pair-7b-dual8",
        target_repo="google/gemma-7b-it",
        draft_repo="google/gemma-2b-it",
        min_chips=8,
        description="Two v6e-8 VMs: 8 chips target + 8 chips draft (16 total)",
    ),
    "sd-pair-2b": ModelProfile(
        name="sd-pair-2b",
        target_repo="google/gemma-2-2b-it",
        draft_repo="google/gemma-2b-it",
        min_chips=4,
        description="Gemma-2.2B target + Gemma-2B draft (v6e-4: 3+1 split)",
    ),
    "sd-pair": ModelProfile(
        name="sd-pair",
        target_repo="google/gemma-2-2b-it",
        draft_repo="google/gemma-2b-it",
        min_chips=4,
        description="Alias for sd-pair-2b",
    ),
}


def get_profile(name: str | None = None) -> ModelProfile:
    key = name or os.getenv("MODEL_PROFILE", "sd-pair-7b")
    if key not in PROFILES:
        raise KeyError(f"Unknown MODEL_PROFILE={key}. Choose: {list(PROFILES)}")
    return PROFILES[key]


def profile_env_defaults(profile: ModelProfile, gcs_bucket: str | None) -> dict[str, str]:
    """Env key/value pairs for a profile."""
    out: dict[str, str] = {
        "MODEL_PROFILE": profile.name,
    }
    if gcs_bucket:
        prefix = os.getenv("GCS_MODEL_PREFIX", "models")
        out["TARGET_MODEL_PATH"] = profile.target_gcs_path(gcs_bucket, prefix)
        out["DRAFT_MODEL_PATH"] = profile.draft_gcs_path(gcs_bucket, prefix)
    else:
        out["TARGET_MODEL_PATH"] = f"./models/{profile.target_repo.replace('/', '_')}"
        out["DRAFT_MODEL_PATH"] = f"./models/{profile.draft_repo.replace('/', '_')}"
    return out
