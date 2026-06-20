"""动作 -> Executor 的映射层：让手势成为一等公民。

执行器只保留极简原语（tap/swipe/long_press/text/key）；scroll/fling/swipe/drag/back/home
等手势全部在此用 swipe + 时长组合实现。这样执行器对原生 App 的 Shizuku 移植零负担，
手势的丰富度集中在这一纯逻辑层。
"""

from __future__ import annotations

from ..types import Action
from .safety import SafetyError, validate_point

# 手势默认参数
SCROLL_MS = 350
FLING_MS = 120
SWIPE_MS = 300
DRAG_MS = 650
LONG_PRESS_MS = 700


def _scroll_endpoints(direction: str, w: int, h: int) -> tuple[int, int, int, int]:
    """返回 (x1,y1,x2,y2)。direction 指「内容滚动方向」：down=向下看更多。"""
    cx, cy = w // 2, h // 2
    if direction == "up":      # 内容上移已看过的，手指从上往下
        return (cx, int(h * 0.30), cx, int(h * 0.72))
    if direction == "left":
        return (int(w * 0.80), cy, int(w * 0.20), cy)
    if direction == "right":
        return (int(w * 0.20), cy, int(w * 0.80), cy)
    # 默认 down：看下方更多内容，手指从下往上
    return (cx, int(h * 0.72), cx, int(h * 0.30))


def execute_action(executor, action: Action, *, point: tuple[int, int] | None, image_wh: tuple[int, int]) -> str:
    """执行一个动作，返回人类可读描述（用于日志/审计）。

    tap/long_press 由调用方解析好像素 ``point`` 传入；其余手势在此按 image_wh 计算。
    """
    w, h = image_wh
    kind = action.kind

    if kind == "tap":
        if point is None:
            raise SafetyError("tap 缺少坐标")
        executor.tap(*point)
        return f"tap{point}"

    if kind == "long_press":
        if point is None:
            raise SafetyError("long_press 缺少坐标")
        executor.long_press(point[0], point[1], action.duration_ms or LONG_PRESS_MS)
        return f"long_press{point}"

    if kind in ("scroll", "fling"):
        x1, y1, x2, y2 = _scroll_endpoints(action.direction or "down", w, h)
        ms = action.duration_ms or (FLING_MS if kind == "fling" else SCROLL_MS)
        executor.swipe(x1, y1, x2, y2, ms)
        return f"{kind}({action.direction or 'down'})"

    if kind == "swipe":
        if action.from_norm and action.to_norm:
            x1, y1 = validate_point(action.from_norm, image_wh)
            x2, y2 = validate_point(action.to_norm, image_wh)
        else:
            x1, y1, x2, y2 = _scroll_endpoints(action.direction or "up", w, h)
        executor.swipe(x1, y1, x2, y2, action.duration_ms or SWIPE_MS)
        return f"swipe(({x1},{y1})->({x2},{y2}))"

    if kind == "drag":
        if not (action.from_norm and action.to_norm):
            raise SafetyError("drag 需要 from/to 坐标")
        x1, y1 = validate_point(action.from_norm, image_wh)
        x2, y2 = validate_point(action.to_norm, image_wh)
        executor.swipe(x1, y1, x2, y2, action.duration_ms or DRAG_MS)
        return f"drag(({x1},{y1})->({x2},{y2}))"

    if kind == "back":
        executor.back()
        return "back"

    if kind == "home":
        executor.home()
        return "home"

    if kind == "key":
        executor.key(action.key or "KEYCODE_BACK")
        return f"key({action.key or 'KEYCODE_BACK'})"

    if kind == "text":
        executor.text(action.text)
        return f"text({action.text!r})"

    raise SafetyError(f"不可执行的动作类型：{kind}")
