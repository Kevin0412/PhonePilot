"""把感知结果格式化成喂给 LLM 的上下文（纯逻辑）。

两套格式对应「快慢双路」：
- ``format_fast``  仅「编号: 文字」，配 thinking=off、低 token —— 高频默认路径，求效率。
- ``format_rich``  带坐标（像素 + 归一化）、样式、图标的 JSON —— 卡住时升级，可返回坐标/手势。
"""

from __future__ import annotations

import json
from typing import Sequence

from ..types import Observation, Plan, Step


def format_fast(elements) -> str:
    """仅编号 + 文字，跳过空文本。"""
    lines: list[str] = []
    for i, el in enumerate(elements):
        t = (el.text or "").strip()
        if not t:
            continue
        lines.append(f"{i}: {t}")
    return "\n".join(lines)


def format_rich(elements, image_w: int, image_h: int) -> str:
    """结构化 JSON：含像素中心、归一化中心、样式、类型（含图标）。"""
    w = max(1, image_w)
    h = max(1, image_h)
    arr = []
    for i, el in enumerate(elements):
        cx, cy = el.center
        arr.append(
            {
                "index": i,
                "text": el.text,
                "center": [cx, cy],
                "center_norm": [round(cx / w, 4), round(cy / h, 4)],
                "style": el.style,
                "kind": el.kind,
            }
        )
    return json.dumps(arr, ensure_ascii=False)


def format_task_state(plan: Plan, step: Step, step_idx: int) -> str:
    """当前任务/子目标描述，置于元素列表之前。"""
    targets = "、".join(step.target_markers) if step.target_markers else "（未指定，自行判断）"
    return (
        f"总任务：{plan.task}\n"
        f"当前子目标（第 {step_idx + 1}/{len(plan.steps)} 步）：{step.goal}\n"
        f"应点击的候选控件：{targets}\n"
        f"该步完成标志：{('、'.join(step.success_markers)) or '（自行判断）'}"
    )


def build_user_fast(plan: Plan, step: Step, step_idx: int, obs: Observation) -> str:
    return f"{format_task_state(plan, step, step_idx)}\n\n当前屏幕元素（编号: 文字）：\n{format_fast(obs.elements)}"


def build_user_rich(plan: Plan, step: Step, step_idx: int, obs: Observation) -> str:
    return (
        f"{format_task_state(plan, step, step_idx)}\n\n"
        f"屏幕尺寸：{obs.image_w}x{obs.image_h}\n"
        f"当前屏幕元素（JSON，含像素与归一化坐标、样式、类型）：\n"
        f"{format_rich(obs.elements, obs.image_w, obs.image_h)}"
    )


def build_user_vision(plan: Plan, step: Step, step_idx: int) -> str:
    return (
        f"{format_task_state(plan, step, step_idx)}\n\n"
        "本机 OCR 无法理解当前屏幕，请直接看截图决定下一步，并给出归一化坐标。"
    )
