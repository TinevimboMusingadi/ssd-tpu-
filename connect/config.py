"""Unified environment configuration for SSD-TPU."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class ConnectConfig:
    gcp_project: str | None
    tpu_zone: str | None
    tpu_vm_name: str | None
    tpu_slice_chips: int | None
    ssh_host: str | None
    ssh_user: str | None
    jax_platforms: str | None
    gcs_bucket: str | None
    gcs_model_prefix: str
    hf_token: str | None
    model_profile: str
    target_model_path: str | None
    draft_model_path: str | None
    sharding_backend: str
    use_toy_model: bool
    tpu_topology: str
    tpu_role: str
    target_vm_name: str | None
    draft_vm_name: str | None

    @classmethod
    def from_env(cls, env_path: str | Path | None = None) -> ConnectConfig:
        if env_path is not None:
            load_dotenv(env_path)
        else:
            load_dotenv()

        chips = os.getenv("TPU_SLICE_CHIPS")
        return cls(
            gcp_project=os.getenv("GCP_PROJECT"),
            tpu_zone=os.getenv("TPU_ZONE"),
            tpu_vm_name=os.getenv("TPU_VM_NAME"),
            tpu_slice_chips=int(chips) if chips else None,
            ssh_host=os.getenv("TPU_SSH_HOST"),
            ssh_user=os.getenv("TPU_SSH_USER"),
            jax_platforms=os.getenv("JAX_PLATFORMS"),
            gcs_bucket=os.getenv("GCS_BUCKET"),
            gcs_model_prefix=os.getenv("GCS_MODEL_PREFIX", "models"),
            hf_token=os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN"),
            model_profile=os.getenv("MODEL_PROFILE", "sd-pair-7b"),
            target_model_path=os.getenv("TARGET_MODEL_PATH"),
            draft_model_path=os.getenv("DRAFT_MODEL_PATH"),
            sharding_backend=os.getenv("SSD_SHARDING_BACKEND", "flax"),
            use_toy_model=os.getenv("SSD_USE_TOY_MODEL", "0").lower() in ("1", "true", "yes"),
            tpu_topology=os.getenv("TPU_TOPOLOGY", "single"),
            tpu_role=os.getenv("SSD_TPU_ROLE", "both"),
            target_vm_name=os.getenv("TARGET_VM_NAME"),
            draft_vm_name=os.getenv("DRAFT_VM_NAME"),
        )

    def apply_jax_platforms(self) -> None:
        if self.jax_platforms:
            os.environ["JAX_PLATFORMS"] = self.jax_platforms

    def apply_to_environ(self) -> None:
        """Push config back into os.environ for child processes."""
        mapping = {
            "GCP_PROJECT": self.gcp_project,
            "TPU_ZONE": self.tpu_zone,
            "TPU_VM_NAME": self.tpu_vm_name,
            "TPU_SLICE_CHIPS": str(self.tpu_slice_chips) if self.tpu_slice_chips else None,
            "GCS_BUCKET": self.gcs_bucket,
            "GCS_MODEL_PREFIX": self.gcs_model_prefix,
            "MODEL_PROFILE": self.model_profile,
            "TARGET_MODEL_PATH": self.target_model_path,
            "DRAFT_MODEL_PATH": self.draft_model_path,
            "SSD_SHARDING_BACKEND": self.sharding_backend,
        }
        for key, val in mapping.items():
            if val is not None:
                os.environ[key] = val
