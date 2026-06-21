"""UIAutomator 无障碍控件树解析（纯逻辑，输入是 XML 字符串）。

标准 Android app 的控件树里 ``EditText`` 节点就是输入框（带精确 bounds、focused），
是「真识别」输入框最可靠的途径。注意：部分自绘 app（如 QQ/微信）不暴露 EditText，
此时本模块返回 None，调用方应回退到 OCR/坐标/位置兜底。
"""

from __future__ import annotations

import re
from typing import Optional

from ..types import ScreenElement

_NODE_RE = re.compile(r"<node\b([^>]*?)/?>")
_ATTR_RE = re.compile(r'([\w:-]+)="([^"]*)"')


def parse_nodes(xml: str) -> list[dict]:
    return [dict(_ATTR_RE.findall(m.group(1))) for m in _NODE_RE.finditer(xml or "")]


def _bounds(value: str) -> Optional[tuple[int, int, int, int]]:
    nums = re.findall(r"-?\d+", value or "")
    if len(nums) >= 4:
        x1, y1, x2, y2 = (int(n) for n in nums[:4])
        return (x1, y1, x2, y2)
    return None


def _center(bounds: tuple[int, int, int, int]) -> tuple[int, int]:
    x1, y1, x2, y2 = bounds
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def input_center(xml: str) -> Optional[tuple[int, int]]:
    """返回输入框（EditText）中心坐标；优先 focused 的。无 EditText 返回 None。"""
    edits = [n for n in parse_nodes(xml) if "EditText" in n.get("class", "")]
    if not edits:
        return None
    edits.sort(key=lambda n: n.get("focused") == "true", reverse=True)
    b = _bounds(edits[0].get("bounds", ""))
    return _center(b) if b else None


def parse_controls(xml: str) -> list[ScreenElement]:
    """把控件树里「有文字/描述的可交互节点」转成 ScreenElement（带精确 bounds）。

    kind: edit=输入框；control=可点击控件（按钮/图标，文字取 text 或 content-desc）。
    """
    out: list[ScreenElement] = []
    for n in parse_nodes(xml):
        cls = n.get("class", "")
        b = _bounds(n.get("bounds", ""))
        if not b:
            continue
        is_edit = "EditText" in cls
        desc = (n.get("content-desc") or "").strip()
        label = (n.get("text") or desc or "").strip()
        clickable = n.get("clickable") == "true"
        # 收纳：输入框 / 可点带文字控件 / 任何带 content-desc 的节点(图标按钮、系统选择器选项等)
        if not is_edit and not (clickable and label) and not desc:
            continue
        x1, y1, x2, y2 = b
        box = ((x1, y1), (x2, y1), (x2, y2), (x1, y2))
        out.append(ScreenElement(text=label, score=1.0, box=box, style="normal", kind="edit" if is_edit else "control"))
    return out
