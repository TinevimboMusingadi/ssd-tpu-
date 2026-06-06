"""SSD-TPU live demo TUI — four panels streaming decoded tokens."""

from __future__ import annotations

import argparse
import threading

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Footer, Header, Static

from connect.mesh_allocator import allocate_meshes
from connect.slice_probe import probe_devices
from jax_ssd.config import DecodeMode
from jax_ssd.llm import LLM
from jax_ssd.sampling_params import SamplingParams
from tui.panels import AlgorithmPanel


class SSDTpuApp(App):
    CSS = """
    AlgorithmPanel {
        width: 1fr;
        height: 1fr;
        border: solid #374151;
        padding: 1;
        overflow-y: auto;
    }
    #status-bar {
        height: 3;
        background: #1a1814;
        padding: 1;
    }
    #prompt-bar {
        height: 3;
        padding: 1;
        border-top: solid #374151;
    }
    """

    MODES = ["ar", "sd", "ssd", "instance"]

    def __init__(self, prompt: str, max_tokens: int = 48) -> None:
        super().__init__()
        self.prompt_text = prompt
        self.max_tokens = max_tokens
        self.prompt_ids = [ord(c) % 64 for c in prompt]

    def compose(self) -> ComposeResult:
        topo = probe_devices()
        alloc = allocate_meshes()
        yield Header()
        yield Static(
            f"SSD-TPU Live Demo │ {topo.summary()} │ {alloc.summary()}",
            id="status-bar",
        )
        with Horizontal():
            for mode in self.MODES:
                yield AlgorithmPanel(mode, id=f"panel-{mode}")
        yield Static(f"Prompt: {self.prompt_text}", id="prompt-bar")
        yield Footer()

    def on_mount(self) -> None:
        for mode in self.MODES:
            thread = threading.Thread(
                target=self._run_mode,
                args=(mode,),
                daemon=True,
            )
            thread.start()

    def _run_mode(self, mode: str) -> None:
        panel: AlgorithmPanel = self.query_one(f"#panel-{mode}", AlgorithmPanel)
        llm = LLM.from_mode(mode)

        def on_token(tok: int, decoded: str) -> None:
            self.call_from_thread(panel.on_token, tok, decoded)

        sampling = SamplingParams(max_new_tokens=self.max_tokens)
        llm.generate([self.prompt_ids], sampling, on_token=on_token)
        llm.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(description="SSD-TPU live TUI")
    parser.add_argument(
        "--prompt",
        default="Refactor getUserById to fetchUserById preserving behavior",
    )
    parser.add_argument("--max-tokens", type=int, default=48)
    args = parser.parse_args()
    SSDTpuApp(prompt=args.prompt, max_tokens=args.max_tokens).run()


if __name__ == "__main__":
    main()
