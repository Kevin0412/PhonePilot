"""动作前安全校验（移植参考项目 validate_target，泛化坐标动作）。

- ``validate_action``  元素类动作：index 越界、index/text 错位（防点错）、黑名单三重校验。
- ``validate_point``   坐标/视觉动作：归一化 -> 像素 + 边界裁剪。
"""

from __future__ import annotations

from typing import Iterable, Sequence

from ..perception.signature import normalize_text
from ..types import Action, ScreenElement


class SafetyError(RuntimeError):
    """动作不安全，应丢弃（不执行）。"""


def validate_action(
    action: Action,
    elements: Sequence[ScreenElement],
    forbidden_markers: Iterable[str] = (),
) -> ScreenElement:
    """校验元素类动作并返回目标元素，不合法抛 SafetyError。"""
    if action.index is None:
        raise SafetyError("元素动作缺少 index")
    if action.index < 0 or action.index >= len(elements):
        raise SafetyError(f"index 越界：{action.index}（共 {len(elements)} 个元素）")

    target = elements[action.index]

    # index/text 一致性：防止 LLM 给的下标与文字对不上而点错位置。
    if action.text:
        a = normalize_text(action.text)
        b = normalize_text(target.text)
        if a and b and a not in b and b not in a:
            raise SafetyError(
                f"index/text 不一致：index={action.index} 实际={target.text!r} 期望={action.text!r}"
            )

    tn = normalize_text(target.text)
    for m in forbidden_markers:
        mn = normalize_text(m)
        if mn and mn in tn:
            raise SafetyError(f"拒绝点击黑名单目标：{target.text!r}")
    return target


def validate_point(point_norm, image_wh: tuple[int, int]) -> tuple[int, int]:
    """归一化坐标 (0~1) -> 像素坐标，并裁剪到屏幕范围内。"""
    if not point_norm or len(point_norm) < 2:
        raise SafetyError("缺少有效坐标")
    w, h = image_wh
    try:
        x = float(point_norm[0])
        y = float(point_norm[1])
    except (TypeError, ValueError) as exc:
        raise SafetyError(f"坐标非数值：{point_norm!r}") from exc
    x = min(max(x, 0.0), 1.0)
    y = min(max(y, 0.0), 1.0)
    return (int(round(x * (max(1, w) - 1))), int(round(y * (max(1, h) - 1))))
