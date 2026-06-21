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
    SCALES = (0.6, 0.75, 0.9, 1.0, 1.1, 1.25, 1.4)

    def __init__(self, template_dir: str | Path, *, threshold: float = 0.72) -> None:
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
                    # 文件名 search_dark.png → 图标名 search（下划线前的第一段作为语义名，
                    # 允许同一图标多个外观变体：search.png / search_dark.png 都输出 [icon:search]）
                    icon_name = png.stem.split("_")[0] if "_" in png.stem else png.stem
                    templates.append((icon_name, tmpl))
        self._templates = templates
        if templates:
            log.info("加载图标模板 %d 个: %s", len(templates), [n for n, _ in templates])
        return templates

    def match(self, image: Any) -> list[ScreenElement]:
        """多尺度模板匹配：每个图标取跨尺度的全局最佳命中（图标一般只出现一次），
        多尺度让模板能跨分辨率/设备适配，不至于因缩放就匹配不上。"""
        templates = self._load()
        if not templates:
            return []
        import cv2  # 懒加载

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gh, gw = gray.shape[:2]
        elements: list[ScreenElement] = []
        for name, tmpl in templates:
            best = None  # (score, x, y, tw, th)
            for s in self.SCALES:
                t = cv2.resize(tmpl, None, fx=s, fy=s) if s != 1.0 else tmpl
                th, tw = t.shape[:2]
                if th >= gh or tw >= gw or th < 8 or tw < 8:
                    continue
                res = cv2.matchTemplate(gray, t, cv2.TM_CCOEFF_NORMED)
                _minv, maxv, _minl, maxl = cv2.minMaxLoc(res)
                if best is None or maxv > best[0]:
                    best = (float(maxv), int(maxl[0]), int(maxl[1]), tw, th)
            if best and best[0] >= self.threshold:
                sc, x, y, tw, th = best
                box = ((x, y), (x + tw, y), (x + tw, y + th), (x, y + th))
                elements.append(ScreenElement(text=f"[icon:{name}]", score=sc, box=box, style="normal", kind="icon"))
        return elements
