"""感知一帧：截图 -> OCR + 图标 -> 结构化元素 + 页面签名 -> Observation。"""

from __future__ import annotations

import logging
from typing import Any

from ..perception.signature import page_signature
from ..types import Observation

log = logging.getLogger(__name__)


def observe(executor, ocr, icons=None, *, with_foreground: bool = True) -> tuple[Any, Observation]:
    image = executor.screencap()
    h = int(image.shape[0])
    w = int(image.shape[1])
    elements = list(ocr.read(image))
    if icons is not None:
        try:
            elements += list(icons.match(image))
        except Exception as exc:  # noqa: BLE001 - 图标失败不应中断感知
            log.debug("图标匹配失败：%s", exc)
    fg = ""
    if with_foreground:
        try:
            fg = executor.foreground()
        except Exception:  # noqa: BLE001
            fg = ""
    obs = Observation(image_w=w, image_h=h, elements=elements, signature=page_signature(elements), foreground=fg)
    return image, obs
