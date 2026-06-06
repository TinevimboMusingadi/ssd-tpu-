"""Load connection settings from environment and optional .env file."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class ConnectConfig:
    gcp_project: str | None
    tpu_zone: str | None
    ssh_host: str | None
    ssh_user: str | None
    jax_platforms: str | None

    @classmethod
    def from_env(cls, env_path: str | Path | None = None) -> ConnectConfig:
        if env_path is not None:
            load_dotenv(env_path)
        else:
            load_dotenv()
        return cls(
            gcp_project=os.getenv("GCP_PROJECT"),
            tpu_zone=os.getenv("TPU_ZONE"),
            ssh_host=os.getenv("TPU_SSH_HOST"),
            ssh_user=os.getenv("TPU_SSH_USER"),
            jax_platforms=os.getenv("JAX_PLATFORMS"),
        )

    def apply_jax_platforms(self) -> None:
        if self.jax_platforms:
            os.environ["JAX_PLATFORMS"] = self.jax_platforms
