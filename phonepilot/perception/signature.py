"""页面语义签名与元素匹配（纯逻辑，零平台依赖）。

移植并泛化自参考项目 ``screen_reader.py``：
- ``normalize_text``  去空格/标点，做模糊匹配。
- ``page_signature``  剔除时钟/倒计时/电量等易变文本，得到稳定的「页面语义指纹」，
                      用于判断页面是否发生语义变化（而非逐像素比较）。
- ``find_element`` / ``find_candidate`` / ``contains``  在元素列表中按文字定位目标。
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Iterable, Optional, Sequence

from ..types import ScreenElement

# 去除空格与各类标点/分隔符，便于模糊匹配。
_NORMALIZE_RE = re.compile(r"[\s|丨·•,，。、.!！?？:：;；()（）<>《》【】\[\]/\\~`@#\$%\^&\*_=+\"'-]+")

_PURE_DIGITS = re.compile(r"^\d+$")
_CLOCK = re.compile(r"^\d{1,2}:\d{2}(:\d{2})?$")
_PERCENT = re.compile(r"^\d{1,3}%$")
_DECIMAL = re.compile(r"^\d+\.\d+$")
# 易变的时间/单位类文本（状态栏时钟、倒计时单位等）。
_VOLATILE_TOKENS = frozenset(
    {"时", "分", "秒", "秒后", "天", "小时", "分钟", "ago", "am", "pm", "上午", "下午"}
)


def normalize_text(value: object) -> str:
    """归一化文本：去空格标点、转小写英文部分。无效输入返回空串。"""
    if not value:
        return ""
    return _NORMALIZE_RE.sub("", str(value)).strip().lower()


def is_volatile(text_norm: str) -> bool:
    """判断归一化后的文本是否为「易变」内容（不应进入页面签名）。"""
    if not text_norm:
        return True
    if _PURE_DIGITS.match(text_norm):
        return True
    if _CLOCK.match(text_norm):
        return True
    if _PERCENT.match(text_norm):
        return True
    if _DECIMAL.match(text_norm):
        return True
    if text_norm in _VOLATILE_TOKENS:
        return True
    return False


def page_signature(elements: Sequence[ScreenElement]) -> str:
    """生成稳定的页面语义指纹。

    规则：剔除易变文本（纯数字/时钟/电量%/单位）、把含数字的「开售/倒计时」类
    归一化为占位符；保留 ``style:文本``；排序后拼接以抵抗 OCR 顺序抖动。
    """
    stable: list[str] = []
    for el in elements:
        t = normalize_text(el.text)
        if is_volatile(t):
            continue
        if ("开售" in t or "倒计时" in t or "countdown" in t) and re.search(r"\d", t):
            stable.append("@countdown")
            continue
        kind_prefix = "icon" if el.kind == "icon" else el.style
        stable.append(f"{kind_prefix}:{t}")
    # 排序使签名对元素出现顺序不敏感（同一页两帧 OCR 顺序可能不同）。
    stable.sort()
    return "\n".join(stable)


def contains(elements: Sequence[ScreenElement], markers: Iterable[str]) -> Optional[str]:
    """任一标志词在任一元素中出现（归一化子串匹配）则返回该标志词原文，否则 None。"""
    norm = [normalize_text(el.text) for el in elements]
    for marker in markers or ():
        m = normalize_text(marker)
        if not m:
            continue
        for t in norm:
            if m and m in t:
                return marker
    return None


def _rejected(text_norm: str, reject_norms: Sequence[str]) -> bool:
    return any(r and r in text_norm for r in reject_norms)


def find_element(
    elements: Sequence[ScreenElement],
    target: str,
    *,
    reject_markers: Iterable[str] = (),
    fuzzy_min: float = 0.62,
    min_y: int = 0,
) -> Optional[tuple[int, ScreenElement]]:
    """在元素中定位 ``target``：精确 → 包含（双向）→ 模糊（SequenceMatcher）。

    返回 ``(index, element)`` 或 None。``reject_markers`` 命中者跳过（如「预填抢票信息」）。
    ``min_y``：跳过中心 y 小于该像素值的元素（用于避开顶部状态栏/搜索框的同名回显）。
    """
    tgt = normalize_text(target)
    if not tgt:
        return None
    rejects = [normalize_text(r) for r in reject_markers]

    def skip(el: ScreenElement, t: str) -> bool:
        return not t or _rejected(t, rejects) or (min_y and el.center[1] < min_y)

    # pass 1: 精确
    for i, el in enumerate(elements):
        t = normalize_text(el.text)
        if skip(el, t):
            continue
        if t == tgt:
            return (i, el)
    # pass 2: 包含（任一方向）
    for i, el in enumerate(elements):
        t = normalize_text(el.text)
        if skip(el, t):
            continue
        if tgt in t or t in tgt:
            return (i, el)
    # pass 3: 模糊
    best_ratio = 0.0
    best: Optional[tuple[int, ScreenElement]] = None
    for i, el in enumerate(elements):
        t = normalize_text(el.text)
        if skip(el, t):
            continue
        ratio = SequenceMatcher(None, tgt, t).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best = (i, el)
    if best is not None:
        # 短目标(≤2 字)用更严格阈值，避免「立即购买」误配「立即抢购」这类高相似异义词
        threshold = max(0.85, fuzzy_min) if len(tgt) <= 2 else fuzzy_min
        if best_ratio >= threshold:
            return best
    return None


def find_candidate(
    elements: Sequence[ScreenElement],
    markers: Iterable[str],
    *,
    reject_markers: Iterable[str] = (),
    fuzzy_min: float = 0.62,
    min_y: int = 0,
) -> Optional[tuple[int, ScreenElement, str]]:
    """按 ``markers`` 顺序找第一个命中的元素，返回 ``(index, element, 命中的marker)``。"""
    for marker in markers or ():
        hit = find_element(elements, marker, reject_markers=reject_markers, fuzzy_min=fuzzy_min, min_y=min_y)
        if hit is not None:
            return (hit[0], hit[1], marker)
    return None
