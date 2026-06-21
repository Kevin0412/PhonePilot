"""感知一帧：截图 -> OCR + 图标 + CV画框 + 控件树 融合 -> Observation。

把多源感知打包给 LLM 决策（LLM 只选 index，精确坐标来自各源包围盒，不需 LLM 吐坐标）：
- OCR：文字 + 包围盒（最可靠的文字标签）
- CV 画框：检出有对比的非文字控件（图标/按钮），补 OCR 看不到的
- 控件树(UIAutomator)：标准 app 的精确控件（EMUI 等自绘 ROM 拿不到，作可选）
页面签名只用文字元素，保持稳定（CV/控件框数量帧间可能抖动，不进签名）。
"""

from __future__ import annotations

import logging
from typing import Any

from ..perception.signature import normalize_text, page_signature
from ..types import Observation, ScreenElement

log = logging.getLogger(__name__)


def _center_in(center, box4) -> bool:
    cx, cy = center
    x1, y1, x2, y2 = box4
    return x1 <= cx <= x2 and y1 <= cy <= y2


def observe(
    executor,
    ocr,
    icons=None,
    *,
    with_foreground: bool = True,
    with_cv: bool = False,
    with_ui: bool = False,
    cv_max_area_ratio: float = 0.06,
) -> tuple[Any, Observation]:
    image = executor.screencap()
    h = int(image.shape[0])
    w = int(image.shape[1])

    elements: list[ScreenElement] = list(ocr.read(image))
    if icons is not None:
        try:
            elements += list(icons.match(image))
        except Exception as exc:  # noqa: BLE001
            log.debug("图标匹配失败：%s", exc)

    # 页面签名只用文字/图标元素，保持稳定
    signature = page_signature(elements)

    # CV 画框：加入「不与已有文字重叠、且控件大小」的非文字控件（图标/按钮）
    if with_cv:
        try:
            from ..perception import cv_controls

            existing = [e.center for e in elements]
            for x1, y1, x2, y2 in cv_controls.detect_boxes(image):
                if (x2 - x1) * (y2 - y1) > cv_max_area_ratio * w * h:
                    continue
                box = (x1, y1, x2, y2)
                if any(_center_in(c, box) for c in existing):
                    continue  # 已被 OCR/图标覆盖
                poly = ((x1, y1), (x2, y1), (x2, y2), (x1, y2))
                elements.append(ScreenElement(text="[控件]", score=1.0, box=poly, style="normal", kind="control"))
        except Exception as exc:  # noqa: BLE001
            log.debug("CV 画框失败：%s", exc)

    # 控件树：加入与已有不重叠的精确控件（标准 app 有效）
    if with_ui:
        try:
            from ..perception import uia

            for ctrl in uia.parse_controls(executor.ui_dump()):
                if any(_center_in(ctrl.center, e.bounds) or _center_in(e.center, ctrl.bounds) for e in elements):
                    continue
                elements.append(ctrl)
        except Exception as exc:  # noqa: BLE001
            log.debug("控件树解析失败：%s", exc)

    fg = ""
    if with_foreground:
        try:
            fg = executor.foreground()
        except Exception:  # noqa: BLE001
            fg = ""
    return image, Observation(image_w=w, image_h=h, elements=elements, signature=signature, foreground=fg)
