"""基于图像的控件检测（CV，无需模型）。

OCR 看不见无文字控件、UIAutomator 在部分自绘 ROM(如 EMUI) 又拿不到 EditText，
此模块直接从截图用 OpenCV 把「常用控件」框出来。当前聚焦最有价值、最稳的一个：
聊天/搜索的**输入框**——底部一条又宽又矮的圆角矩形。cv2 仅在此懒加载。
"""

from __future__ import annotations

from typing import Any, Optional


def detect_input_box(
    image: Any,
    *,
    bottom_ratio: float = 0.80,
    min_w_ratio: float = 0.40,
    min_h_ratio: float = 0.018,
    max_h_ratio: float = 0.12,
    max_w_ratio: float = 0.98,
) -> Optional[tuple[int, int]]:
    """检测底部输入栏，返回其中心像素坐标；找不到返回 None。

    思路：在屏幕底部条带里找「宽度大、高度适中、不顶满全宽」的矩形轮廓，取最靠下的较宽者
    （输入框通常在该条带最上方一行、且比图标行宽）。
    """
    import cv2
    import numpy as np

    h, w = image.shape[:2]
    y0 = int(h * bottom_ratio)
    strip = image[y0:h]
    if strip.size == 0:
        return None
    gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 40, 120)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates: list[tuple[int, int, int, int]] = []
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        if cw < w * min_w_ratio or cw > w * max_w_ratio:
            continue
        if ch < h * min_h_ratio or ch > h * max_h_ratio:
            continue
        candidates.append((x, y, cw, ch))
    if not candidates:
        return None
    # 输入框一般在底部条带里最宽、且偏上的一行；优先最宽，再偏上
    candidates.sort(key=lambda r: (r[2], -r[1]), reverse=True)
    x, y, cw, ch = candidates[0]
    return (x + cw // 2, y0 + y + ch // 2)
