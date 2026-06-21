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


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / float(area_a + area_b - inter)


def detect_boxes(
    image: Any,
    *,
    min_wh: int = 26,
    min_area_ratio: float = 0.0006,
    max_area_ratio: float = 0.30,
    max_w_ratio: float = 0.97,
    iou_dedup: float = 0.6,
) -> list[tuple[int, int, int, int]]:
    """从截图检测「候选控件」的包围盒(画框)，返回 (x1,y1,x2,y2) 列表。

    用边缘+轮廓找有视觉结构的矩形(按钮/图标/卡片/带边输入框)。对**有对比**的控件有效；
    近白无对比的(如 QQ 空输入框)框不出——那类靠 UIAutomator/位置兜底。
    """
    import cv2
    import numpy as np

    h, w = image.shape[:2]
    area = w * h
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(g, 30, 110)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    raw: list[tuple[int, int, int, int]] = []
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        if cw < min_wh or ch < min_wh:
            continue
        if cw > w * max_w_ratio and ch > h * 0.5:
            continue  # 整屏容器
        a = cw * ch
        if a < area * min_area_ratio or a > area * max_area_ratio:
            continue
        raw.append((x, y, x + cw, y + ch))

    # 按面积大到小做 IoU 去重（保留较大、合并重叠）
    raw.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
    kept: list[tuple[int, int, int, int]] = []
    for b in raw:
        if all(_iou(b, k) < iou_dedup for k in kept):
            kept.append(b)
    return kept

