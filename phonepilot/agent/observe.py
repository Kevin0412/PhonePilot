"""感知一帧：截图 -> OCR + 图标 + 控件树 + CV画框 融合 -> Observation。

把多源感知打包给 LLM 决策（LLM 只选 index，精确坐标来自各源包围盒，不需 LLM 吐坐标）。
融合顺序按「标签质量」排，后者只补前者没覆盖的区域：
- OCR：文字 + 包围盒（最可靠的文字标签）
- 图标：模板命中的常见图标（搜索/返回…），带语义 [icon:xxx]
- 控件树(UIAutomator)：带 content-desc/text 的精确控件，首选「画框」源；
  还能看到截图拍不到的系统浮层（如华为主/分身选择器）。EMUI 等自绘 ROM 拿不到时自然为空。
- CV 画框：兜底，仅补上述都没覆盖的无标签控件；信息量低，去重后数量封顶防淹没。
页面签名只用文字/图标元素，保持稳定（CV/控件框数量帧间可能抖动，不进签名）。
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
    ui_detect=None,
    vl_caption: bool = False,
    with_foreground: bool = True,
    with_cv: bool = False,
    with_ui: bool = False,
    cv_max_area_ratio: float = 0.06,
    cv_max_boxes: int = 12,
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

    # 控件树先于 CV：它给「带标签(content-desc/text)的精确控件」，是首选的「画框」源
    # （还能看到截图拍不到的系统浮层，如华为主/分身选择器）。与已有文字/图标不重叠才加入。
    if with_ui:
        try:
            from ..perception import uia

            for ctrl in uia.parse_controls(executor.ui_dump()):
                if any(_center_in(ctrl.center, e.bounds) or _center_in(e.center, ctrl.bounds) for e in elements):
                    continue
                elements.append(ctrl)
        except Exception as exc:  # noqa: BLE001
            log.debug("控件树解析失败：%s", exc)

    # UI 检测器（YOLO）开启时替代粗糙 CV 画框——检出质量高、覆盖输入框等低对比控件。
    _used_detector = False
    if ui_detect is not None:
        try:
            for det in ui_detect.detect(image):
                if any(_center_in(det.center, e.bounds) or _center_in(e.center, det.bounds) for e in elements):
                    continue
                elements.append(det)
            _used_detector = True
        except Exception as exc:  # noqa: BLE001
            log.debug("UI 检测器失败，回退粗糙 CV：%s", exc)

    # 粗糙 CV 画框：仅当无检测器（或检测器失败）时才启用。
    if with_cv and not _used_detector:
        try:
            from ..perception import cv_controls

            kept = 0
            dropped = 0
            for x1, y1, x2, y2 in cv_controls.detect_boxes(image):
                if (x2 - x1) * (y2 - y1) > cv_max_area_ratio * w * h:
                    continue
                box = (x1, y1, x2, y2)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                # 与任意已有元素（OCR/图标/控件树）相互包含即视为重复
                if any(_center_in((cx, cy), e.bounds) or _center_in(e.center, box) for e in elements):
                    continue
                if kept >= cv_max_boxes:
                    dropped += 1
                    continue
                poly = ((x1, y1), (x2, y1), (x2, y2), (x1, y2))
                elements.append(ScreenElement(text="[控件]", score=1.0, box=poly, style="normal", kind="control"))
                kept += 1
            if dropped:
                log.debug("CV 画框超上限(%d)，丢弃 %d 个无标签控件框", cv_max_boxes, dropped)
        except Exception as exc:  # noqa: BLE001
            log.debug("CV 画框失败：%s", exc)

    # VL 标注：给无标签 [控件] 加语义（"搜索"、"返回"等），一次调用标注全部
    if vl_caption and any(e.text in ("[控件]", "") and e.kind in ("ui", "control") for e in elements):
        try:
            from ..perception.vl_caption import caption_controls

            elements = caption_controls(image, elements)
        except Exception as exc:  # noqa: BLE001
            log.debug("VL 标注失败：%s", exc)

    fg = ""
    if with_foreground:
        try:
            fg = executor.foreground()
        except Exception:  # noqa: BLE001
            fg = ""
    return image, Observation(image_w=w, image_h=h, elements=elements, signature=signature, foreground=fg)
