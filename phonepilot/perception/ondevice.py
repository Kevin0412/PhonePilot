"""On-device OCR 适配器槽位（扩展点）。

原生 App 阶段在此接 ML Kit / onnxruntime-android / MNN 等设备端 OCR，实现 OcrEngine
接口、产出同样的 ScreenElement，即可平滑替换 RapidOCR。MVP 仅占位。
"""

from __future__ import annotations

from typing import Any

from ..types import ScreenElement
from .base import OcrEngine


class OnDeviceOcrEngine(OcrEngine):  # pragma: no cover - 扩展点
    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            "On-device OCR 尚未实现。原生 App 阶段在此接 ML Kit / onnxruntime-android，"
            "实现 read() 返回 ScreenElement 列表即可。"
        )

    def read(self, image: Any, *, scale: float = 1.0) -> list[ScreenElement]:
        raise NotImplementedError
