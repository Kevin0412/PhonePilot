"""RapidOCR 适配器（默认 OCR 引擎）。

移植自参考项目 ``screen_reader.py``：RapidOCR(onnxruntime) 识别 + HSV 样式检测
（主按钮 / 选中态）。cv2 与 rapidocr 仅在此模块、且在首次实例化时懒加载，
因此核心逻辑与测试无需安装这些重型依赖。
"""

from __future__ import annotations

from typing import Any

from ..types import ScreenElement
from .base import OcrEngine


class RapidOcrEngine(OcrEngine):
    def __init__(
        self,
        *,
        min_score: float = 0.55,
        use_cuda: bool = False,
        primary_hsv_low: tuple[int, int, int] = (150, 55, 145),
        primary_hsv_high: tuple[int, int, int] = (179, 255, 255),
        primary_ratio: float = 0.28,
        selected_ratio: float = 0.015,
    ) -> None:
        self.min_score = min_score
        self.primary_hsv_low = primary_hsv_low
        self.primary_hsv_high = primary_hsv_high
        self.primary_ratio = primary_ratio
        self.selected_ratio = selected_ratio
        # 懒加载重型依赖
        try:
            from rapidocr_onnxruntime import RapidOCR  # type: ignore
        except Exception as exc:  # pragma: no cover - 仅真机路径
            raise RuntimeError(
                "RapidOcrEngine 需要 rapidocr_onnxruntime；请 `pip install -e \".[device]\"`"
            ) from exc
        self._engine = RapidOCR()

    def read(self, image: Any, *, scale: float = 1.0) -> list[ScreenElement]:
        import cv2  # 懒加载
        import numpy as np

        work = image
        if scale and scale != 1.0:
            work = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        inv = (1.0 / scale) if scale else 1.0

        result, _ = self._engine(work)
        items: list[ScreenElement] = []
        for raw_box, text, score in result or []:
            if not text or float(score) < self.min_score:
                continue
            pts = [(int(round(p[0] * inv)), int(round(p[1] * inv))) for p in raw_box]
            box = tuple(pts)
            style = self._detect_style(image, box, np, cv2)
            items.append(
                ScreenElement(text=str(text), score=float(score), box=box, style=style, kind="text")
            )
        return items

    def _detect_style(self, image: Any, box, np, cv2) -> str:
        """根据元素周边像素的 HSV 占比判断是否为粉色主按钮 / 选中态。"""
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        x1, y1, x2, y2 = max(0, min(xs)), max(0, min(ys)), max(xs), max(ys)
        if x2 <= x1 or y2 <= y1:
            return "normal"
        region = image[y1:y2, x1:x2]
        if region.size == 0:
            return "normal"
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array(self.primary_hsv_low), np.array(self.primary_hsv_high))
        ratio = float(np.count_nonzero(mask)) / float(mask.size or 1)
        if ratio >= self.primary_ratio:
            return "primary_button"
        if ratio >= self.selected_ratio:
            return "selected"
        return "normal"
