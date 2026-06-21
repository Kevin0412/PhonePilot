"""VL 标注：给检测器出的无标签控件框加语义（全图+编号画框 → VL → 标签）。

一次 VL 调用标注一批控件。只对 text="[控件]" 的元素做标注，已有文字标签的跳过。
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
from typing import Any, Optional

from ..types import ScreenElement

log = logging.getLogger(__name__)

_PROMPT = (
    "图中红色编号框标出了可交互UI控件。"
    "回复JSON：key是编号(字符串)，value是控件功能名(2-4字，如\"搜索\"\"返回\"\"添加\"\"发送\"\"菜单\"\"语音\"\"表情\")。"
    "只输出一个JSON对象，不要解释。"
)


def caption_controls(
    image: Any,
    elements: list[ScreenElement],
    *,
    base_url: str = "https://token-plan-cn.xiaomimimo.com/v1",
    model: str = "mimo-v2.5",
    env_key: str = "MIMO_API_KEY",
    timeout: float = 20.0,
    max_long_side: int = 960,
) -> list[ScreenElement]:
    """给无标签控件加 VL 语义标签，返回更新后的完整元素列表（原地替换）。"""
    import cv2
    import numpy as np

    h_img, w_img = image.shape[:2]
    min_area = w_img * h_img * 0.0005  # 太小的框跳过（状态栏图标等）
    unlabeled = [(i, e) for i, e in enumerate(elements)
                 if e.text in ("[控件]", "") and e.kind in ("ui", "control")
                 and (e.bounds[2] - e.bounds[0]) * (e.bounds[3] - e.bounds[1]) > min_area]
    if not unlabeled:
        return elements
    if len(unlabeled) > 15:
        unlabeled = unlabeled[:15]  # VL token 有限，最多标 15 个

    api_key = os.environ.get(env_key, "")
    if not api_key:
        return elements

    vis = image.copy()
    idx_map: dict[int, int] = {}  # 画框编号 → elements 下标
    for draw_idx, (elem_idx, e) in enumerate(unlabeled):
        x1, y1, x2, y2 = e.bounds
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(vis, str(draw_idx), (x1 + 2, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        idx_map[draw_idx] = elem_idx

    scale = max_long_side / max(h_img, w_img)
    small = cv2.resize(vis, (int(w_img * scale), int(h_img * scale)))
    _, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 75])
    b64 = base64.b64encode(buf).decode()

    from ..planner.llm_base import http_post_json

    try:
        resp = http_post_json(
            base_url, "/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            payload={
                "model": model,
                "messages": [{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text", "text": _PROMPT},
                ]}],
                "max_tokens": 300,
            },
            timeout=timeout,
        )
    except Exception as exc:
        log.debug("VL 标注请求失败：%s", exc)
        return elements

    msg = resp.get("choices", [{}])[0].get("message", {})
    text = msg.get("content", "") or msg.get("reasoning_content", "")

    labels = _extract_labels(text)
    if not labels:
        log.debug("VL 标注未返回可解析标签，原文：%s", text[:200])
        return elements

    updated = list(elements)
    applied = 0
    for draw_idx, label in labels.items():
        elem_idx = idx_map.get(draw_idx)
        if elem_idx is None or elem_idx >= len(updated):
            continue
        old = updated[elem_idx]
        updated[elem_idx] = ScreenElement(
            text=f"[{label}]", score=old.score, box=old.box, style=old.style, kind=old.kind,
        )
        applied += 1
    if applied:
        log.info("VL 标注 %d/%d 个控件", applied, len(unlabeled))
    return updated


def _extract_labels(text: str) -> dict[int, str]:
    """从 VL 回复（JSON 或推理文本）提取 编号→标签 映射。"""
    # 先尝试直接 JSON
    for m in re.finditer(r'\{[^{}]+\}', text):
        try:
            d = json.loads(m.group())
            if d and all(isinstance(v, str) for v in d.values()):
                return {int(k): v for k, v in d.items() if k.isdigit()}
        except (json.JSONDecodeError, ValueError):
            pass
    # 回退：从推理文本提取 编号N ... 控件名"XXX"
    labels: dict[int, str] = {}
    for m in re.finditer(r'编号?\s*(\d+)[^""\n]*?控件名?\s*["""]([^"""\n]{1,8})["""]', text):
        labels[int(m.group(1))] = m.group(2)
    if labels:
        return labels
    # 再回退：N: "XXX" 或 N："XXX"
    for m in re.finditer(r'(\d+)\s*[:：]\s*["""]([^"""\n]{1,8})["""]', text):
        labels[int(m.group(1))] = m.group(2)
    return labels
