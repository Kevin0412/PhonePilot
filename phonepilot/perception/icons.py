"""图标匹配器。

MVP 提供一个基于模板匹配的最小可用实现 ``TemplateIconMatcher``：从 ``template_dir``
读取常见图标 PNG（文件名即图标名，如 ``back.png`` / ``menu.png`` / ``search.png``），
用 cv2.matchTemplate 在截图中定位，命中者产出 ``[icon:<name>]`` 元素。

未放模板时它返回空列表（等价于关闭）。后续可替换为轻量检测模型；未知图标交由视觉
LLM 兜底。cv2 仅在此懒加载。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ..types import ScreenElement
from .base import IconMatcher, NoopIconMatcher  # re-export NoopIconMatcher

__all__ = ["TemplateIconMatcher", "NoopIconMatcher"]

log = logging.getLogger(__name__)


class TemplateIconMatcher(IconMatcher):
    def __init__(self, template_dir: str | Path, *, threshold: float = 0.82) -> None:
        self.template_dir = Path(template_dir)
        self.threshold = threshold
        self._templates: list[tuple[str, Any]] | None = None  # 懒加载

    def _load(self) -> list[tuple[str, Any]]:
        if self._templates is not None:
            return self._templates
        templates: list[tuple[str, Any]] = []
        if self.template_dir.is_dir():
            try:
                import cv2  # 懒加载
            except Exception as exc:  # pragma: no cover
                log.warning("图标匹配需要 opencv：%s", exc)
                self._templates = []
                return self._templates
            for png in sorted(self.template_dir.glob("*.png")):
                tmpl = cv2.imread(str(png), cv2.IMREAD_GRAYSCALE)
                if tmpl is not None and tmpl.size:
                    templates.append((png.stem, tmpl))
        self._templates = templates
        if templates:
            log.info("加载图标模板 %d 个: %s", len(templates), [n for n, _ in templates])
        return templates

    def match(self, image: Any) -> list[ScreenElement]:
        templates = self._load()
        if not templates:
            return []
        import cv2  # 懒加载
        import numpy as np

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elements: list[ScreenElement] = []
        for name, tmpl in templates:
            th, tw = tmpl.shape[:2]
            if gray.shape[0] < th or gray.shape[1] < tw:
                continue
            res = cv2.matchTemplate(gray, tmpl, cv2.TM_CCOEFF_NORMED)
            loc = np.where(res >= self.threshold)
            seen: list[tuple[int, int]] = []
            for y, x in zip(*loc):
                # 简单去重：避免同一图标的相邻多次命中
                if any(abs(x - sx) < tw and abs(y - sy) < th for sx, sy in seen):
                    continue
                seen.append((int(x), int(y)))
                box = ((int(x), int(y)), (int(x + tw), int(y)),
                       (int(x + tw), int(y + th)), (int(x), int(y + th)))
                elements.append(
                    ScreenElement(
                        text=f"[icon:{name}]",
                        score=float(res[y, x]),
                        box=box,
                        style="normal",
                        kind="icon",
                    )
                )
        return elements
