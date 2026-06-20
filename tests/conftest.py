"""测试公共辅助。"""

from __future__ import annotations

from phonepilot.types import ScreenElement


def el(text: str, *, center=(540, 1000), style="normal", kind="text", score=0.95) -> ScreenElement:
    return ScreenElement.from_dict(
        {"text": text, "center": list(center), "style": style, "kind": kind, "score": score}
    )
