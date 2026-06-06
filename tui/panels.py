"""TUI panel widgets."""

from __future__ import annotations

from textual.widgets import Static

from tui.theme import THEME
from tui.token_anim import TokenStream


class AlgorithmPanel(Static):
    """Single algorithm column: metrics header + live decoded output."""

    def __init__(self, mode: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self.stream = TokenStream(mode=mode)
        self.border_title = mode.upper()

    def on_token(self, _token_id: int, decoded: str) -> None:
        self.stream.append(decoded if decoded.endswith(" ") else decoded + " ")
        self.refresh_panel()

    def refresh_panel(self) -> None:
        color = THEME.get(self.stream.mode, THEME["text"])
        flash = " bold" if self.stream.should_flash() else ""
        self.update(
            f"[{color}]{self.stream.metrics_line()}[/{color}]{flash}\n"
            f"{self.stream.display_text}"
        )
