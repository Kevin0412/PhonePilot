"""PhonePilot 公共数据类型。

纯逻辑、零平台依赖（不 import numpy/cv2/http）：屏幕图像在执行/感知层以
``numpy.ndarray`` 传递，但本模块只描述其结构化结果与决策，故无需引入这些库。
这使本模块可被任何环境零依赖 import，也是后续 Kotlin/Chaquopy 移植的接口规范。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# --------------------------------------------------------------------------- #
# 风险等级
# --------------------------------------------------------------------------- #
class RiskLevel(Enum):
    """动作风险分级。"""

    LOW = "low"        # 点击/滑动/读取等常规、可逆操作
    MEDIUM = "medium"  # 输入文本、切换开关等有副作用但低危
    HIGH = "high"      # 支付/删除/发送/授权/卸载/转账 —— 默认需人工确认

    @property
    def rank(self) -> int:
        return {"low": 0, "medium": 1, "high": 2}[self.value]

    @classmethod
    def from_str(cls, value: object, default: "RiskLevel" = None) -> "RiskLevel":
        """从任意 LLM/配置取值容错解析，无法识别时回落到 default（默认 LOW）。"""
        default = default or cls.LOW
        if isinstance(value, RiskLevel):
            return value
        if not isinstance(value, str):
            return default
        key = value.strip().lower()
        for level in cls:
            if level.value == key:
                return level
        # 常见别名
        if key in {"safe", "normal", "none"}:
            return cls.LOW
        if key in {"danger", "dangerous", "critical"}:
            return cls.HIGH
        return default

    @classmethod
    def max_of(cls, *levels: "RiskLevel") -> "RiskLevel":
        best = cls.LOW
        for lv in levels:
            if lv.rank > best.rank:
                best = lv
        return best


# --------------------------------------------------------------------------- #
# 动作词汇
# --------------------------------------------------------------------------- #
# 完整手势词汇——手机操作远不止点击。
ACTION_KINDS: frozenset[str] = frozenset(
    {
        "tap",
        "long_press",
        "swipe",
        "scroll",
        "fling",
        "drag",
        "back",
        "home",
        "text",
        "key",
        "launch",   # 命令启动 app（text=包名）
        "intent",   # 命令直达系统页（text=intent action）
        "wait",
        "done",
        "stop",
    }
)

# 引用某个 OCR/图标元素下标、点击其中心的动作。
ELEMENT_ACTIONS: frozenset[str] = frozenset({"tap", "long_press"})

# 导航类手势：执行后不强制断言「页面发生变化」（页内滚动可能不跳页）。
NAV_GESTURES: frozenset[str] = frozenset({"scroll", "fling", "swipe", "drag"})

# 滚动方向 -> 文字（供手势映射层使用）。
SCROLL_DIRECTIONS: frozenset[str] = frozenset({"up", "down", "left", "right"})


# --------------------------------------------------------------------------- #
# 屏幕元素与观测
# --------------------------------------------------------------------------- #
Point = tuple[int, int]
Box = tuple[Point, ...]


@dataclass(frozen=True)
class ScreenElement:
    """一次感知得到的单个屏幕元素（OCR 文本或图标）。"""

    text: str
    score: float = 1.0
    box: Box = ()                 # 4 个角点 ((x1,y1),...)；可为空（仅图标占位）
    style: str = "normal"         # normal | primary_button | selected
    kind: str = "text"            # text | icon

    @property
    def bounds(self) -> tuple[int, int, int, int]:
        """轴对齐包围盒 (min_x, min_y, max_x, max_y)。"""
        if not self.box:
            return (0, 0, 0, 0)
        xs = [p[0] for p in self.box]
        ys = [p[1] for p in self.box]
        return (min(xs), min(ys), max(xs), max(ys))

    @property
    def center(self) -> Point:
        x1, y1, x2, y2 = self.bounds
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "score": round(float(self.score), 4),
            "box": [list(p) for p in self.box],
            "center": list(self.center),
            "style": self.style,
            "kind": self.kind,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ScreenElement":
        """从 mock OCR JSON / 审计记录构造。支持 box 或仅 center（合成小框）。"""
        box_raw = data.get("box")
        if box_raw:
            box = tuple((int(p[0]), int(p[1])) for p in box_raw)
        elif data.get("center"):
            cx, cy = int(data["center"][0]), int(data["center"][1])
            half = int(data.get("half", 8))
            box = ((cx - half, cy - half), (cx + half, cy - half),
                   (cx + half, cy + half), (cx - half, cy + half))
        else:
            box = ()
        return cls(
            text=str(data.get("text", "")),
            score=float(data.get("score", 1.0)),
            box=box,
            style=str(data.get("style", "normal")),
            kind=str(data.get("kind", "text")),
        )


@dataclass
class Observation:
    """某一帧的结构化感知结果。"""

    image_w: int
    image_h: int
    elements: list[ScreenElement] = field(default_factory=list)
    signature: str = ""
    foreground: str = ""


# --------------------------------------------------------------------------- #
# 动作与决策
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Action:
    """一个待执行的原语/手势。

    - 元素类动作（tap/long_press）用 ``index`` 引用 Observation.elements。
    - 坐标/手势类动作用归一化坐标（0~1）：``point`` 或 ``from_norm``/``to_norm``，
      或方向 ``direction``（scroll/fling/swipe）。归一化坐标与设备分辨率解耦，
      便于跨设备与视觉模型输出对接。
    """

    kind: str
    index: Optional[int] = None
    text: str = ""
    point: Optional[tuple[float, float]] = None        # 归一化 (x,y) ∈ [0,1]
    from_norm: Optional[tuple[float, float]] = None     # drag/swipe 起点（归一化）
    to_norm: Optional[tuple[float, float]] = None       # drag/swipe 终点（归一化）
    direction: str = ""                                  # scroll/fling/swipe 方向
    duration_ms: int = 0                                 # 0 表示用映射层默认
    key: str = ""                                        # key 动作的 keycode
    risk: RiskLevel = RiskLevel.LOW
    reason: str = ""

    @property
    def uses_index(self) -> bool:
        return self.kind in ELEMENT_ACTIONS and self.index is not None

    @property
    def is_terminal(self) -> bool:
        return self.kind in {"done", "stop"}

    @property
    def is_nav_gesture(self) -> bool:
        return self.kind in NAV_GESTURES


@dataclass(frozen=True)
class Decision:
    """决策器输出：动作 + 来源（用于日志/审计与优先级裁决）。"""

    action: Action
    source: str = "local"   # local | text-llm-fast | text-llm-rich | vision | planner


# --------------------------------------------------------------------------- #
# 任务规划
# --------------------------------------------------------------------------- #
@dataclass
class Step:
    """Plan 的一个子目标。"""

    goal: str
    success_markers: list[str] = field(default_factory=list)   # 出现即视为本步完成
    target_markers: list[str] = field(default_factory=list)    # 本步应点击的候选文字
    package: str = ""           # 若本步是打开某 app，填包名 -> 命令启动，不点图标
    intent: str = ""            # 若本步可直达某系统页面，填 intent action -> 命令直达（最稳）
    input_text: str = ""        # 若本步是输入文字，填要输入的内容（先聚焦 target，再用输入法打字）
    verify: list[str] = field(default_factory=list)  # 执行本步动作前必须出现的文字（前置校验，全部命中才执行）
    risk: RiskLevel = RiskLevel.LOW

    def to_dict(self) -> dict:
        return {
            "goal": self.goal,
            "success_markers": list(self.success_markers),
            "target_markers": list(self.target_markers),
            "package": self.package,
            "intent": self.intent,
            "input_text": self.input_text,
            "verify": list(self.verify),
            "risk": self.risk.value,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Step":
        return cls(
            goal=str(data.get("goal", "")).strip(),
            success_markers=[str(s) for s in data.get("success_markers", []) if str(s).strip()],
            target_markers=[str(s) for s in data.get("target_markers", []) if str(s).strip()],
            package=str(data.get("package", "")).strip(),
            intent=str(data.get("intent", "")).strip(),
            input_text=str(data.get("input_text", "")),
            verify=[str(s) for s in data.get("verify", []) if str(s).strip()],
            risk=RiskLevel.from_str(data.get("risk")),
        )


@dataclass
class Plan:
    """自然语言任务的结构化执行计划。"""

    task: str
    goal: str = ""
    steps: list[Step] = field(default_factory=list)
    final_success_markers: list[str] = field(default_factory=list)
    overall_risk: RiskLevel = RiskLevel.LOW

    def to_dict(self) -> dict:
        return {
            "task": self.task,
            "goal": self.goal,
            "steps": [s.to_dict() for s in self.steps],
            "final_success_markers": list(self.final_success_markers),
            "overall_risk": self.overall_risk.value,
        }

    @classmethod
    def from_dict(cls, task: str, data: dict) -> "Plan":
        steps = [Step.from_dict(s) for s in data.get("steps", []) if isinstance(s, dict)]
        return cls(
            task=task,
            goal=str(data.get("goal", "")).strip(),
            steps=steps,
            final_success_markers=[
                str(s) for s in data.get("final_success_markers", []) if str(s).strip()
            ],
            overall_risk=RiskLevel.from_str(data.get("overall_risk")),
        )
