"""Mock 执行器 —— 让整条 感知→决策→执行 闭环在**无真机**的 PC 上跑通。

模型：一串 ``MockFrame``（每帧 = 元素列表 + 前台 + 可选图像）。点击/长按默认把游标
推进到下一帧，从而模拟页面跳转；``back`` 回退一帧。每帧合成图像填充不同灰度值，
使 ``wait_for_change`` 能检测到「页面变化」。

配套 ``MockOcrEngine`` 直接返回当前帧的元素（不依赖真实 OCR），因此测试无需 cv2/rapidocr。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np

from ..types import ScreenElement
from .base import Executor


@dataclass
class MockFrame:
    elements: list[ScreenElement] = field(default_factory=list)
    foreground: str = ""
    width: int = 1080
    height: int = 2400
    image: Optional[Any] = None  # 留空则按帧序合成一张可区分的纯色图

    @classmethod
    def from_dict(cls, data: dict) -> "MockFrame":
        els = [ScreenElement.from_dict(e) for e in data.get("elements", [])]
        return cls(
            elements=els,
            foreground=str(data.get("foreground", "")),
            width=int(data.get("width", 1080)),
            height=int(data.get("height", 2400)),
        )


class MockExecutor(Executor):
    def __init__(
        self,
        frames: Sequence[MockFrame],
        *,
        advance_on: Sequence[str] = ("tap", "long_press"),
    ) -> None:
        if not frames:
            raise ValueError("MockExecutor 需要至少一帧")
        self.frames = list(frames)
        self.cursor = 0
        self.advance_on = set(advance_on)
        self.actions: list[dict] = []   # 供测试断言

    # ---- 内部 ---- #
    def current_frame(self) -> MockFrame:
        return self.frames[self.cursor]

    def _synth_image(self) -> Any:
        f = self.current_frame()
        if f.image is not None:
            return f.image
        # 每帧不同填充值，确保相邻帧的 screen_distance 足够大。
        fill = (self.cursor * 53 + 20) % 220 + 10
        return np.full((f.height, f.width, 3), fill, dtype=np.uint8)

    def _record(self, kind: str, **kw: Any) -> None:
        self.actions.append({"kind": kind, "cursor": self.cursor, **kw})
        if kind in self.advance_on:
            self.cursor = min(self.cursor + 1, len(self.frames) - 1)

    # ---- 接口实现 ---- #
    def screencap(self) -> Any:
        return self._synth_image()

    def tap(self, x: int, y: int) -> None:
        self._record("tap", x=int(x), y=int(y))

    def swipe(self, x1: int, y1: int, x2: int, y2: int, ms: int = 300) -> None:
        self._record("swipe", x1=int(x1), y1=int(y1), x2=int(x2), y2=int(y2), ms=int(ms))

    def long_press(self, x: int, y: int, ms: int = 700) -> None:
        self._record("long_press", x=int(x), y=int(y), ms=int(ms))

    def text(self, value: str) -> None:
        self._record("text", value=value)

    def key(self, keycode: str) -> None:
        code = str(keycode).upper()
        if "BACK" in code:
            self.cursor = max(self.cursor - 1, 0)
        self._record("key", keycode=keycode)

    def foreground(self) -> str:
        return self.current_frame().foreground


class MockOcrEngine:
    """返回 MockExecutor 当前帧元素的「OCR 引擎」（用于测试/演示）。"""

    def __init__(self, executor: MockExecutor) -> None:
        self.executor = executor

    def read(self, image: Any, *, scale: float = 1.0) -> list[ScreenElement]:
        return list(self.executor.current_frame().elements)
