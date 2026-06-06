"""SSH tunnel helpers for stable Remote-SSH TPU development."""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass

from connect.config import ConnectConfig


@dataclass
class SSHTunnel:
    """Manage a local port-forward to a TPU VM."""

    host: str
    user: str | None
    local_port: int
    remote_port: int
    _process: subprocess.Popen | None = None

    @classmethod
    def from_config(cls, config: ConnectConfig, local_port: int = 8888) -> SSHTunnel | None:
        if not config.ssh_host:
            return None
        return cls(
            host=config.ssh_host,
            user=config.ssh_user,
            local_port=local_port,
            remote_port=8888,
        )

    @property
    def target(self) -> str:
        return f"{self.user}@{self.host}" if self.user else self.host

    def start(self) -> None:
        if self._process is not None:
            return
        cmd = [
            "ssh",
            "-N",
            "-L",
            f"{self.local_port}:localhost:{self.remote_port}",
            self.target,
            "-o",
            "ServerAliveInterval=60",
            "-o",
            "ServerAliveCountMax=3",
            "-o",
            "ExitOnForwardFailure=yes",
        ]
        self._process = subprocess.Popen(cmd)

    def stop(self) -> None:
        if self._process is not None:
            self._process.terminate()
            self._process.wait(timeout=5)
            self._process = None

    def is_alive(self) -> bool:
        if self._process is None:
            return False
        return self._process.poll() is None

    def ensure_connected(self, retries: int = 3, delay_s: float = 2.0) -> bool:
        for _ in range(retries):
            if self.is_alive():
                return True
            try:
                self.start()
                time.sleep(delay_s)
            except OSError:
                time.sleep(delay_s)
        return self.is_alive()
