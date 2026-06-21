"""风险闸门：动作分级 + 高风险/不可校验动作的人工确认。

落实「高风险动作默认需人工确认」。判级综合三处：LLM 自评风险、命中高风险标志词、
动作类型（文本输入至少 medium）。``forbidden_markers`` 命中则直接拒绝（永不自动点击）。

``confirm_fn`` 可注入（默认 ``input``），便于单测与非交互环境。
"""

from __future__ import annotations

import logging
from typing import Callable, Iterable, Optional

from ..perception.signature import normalize_text
from ..types import Action, RiskLevel

log = logging.getLogger(__name__)

_YES = {"y", "yes", "是", "确认", "确定", "ok", "好", "continue", "c"}


class RiskGate:
    def __init__(
        self,
        *,
        auto_confirm_low: bool = True,
        auto_confirm_medium: bool = False,
        auto_confirm_all: bool = False,
        confirm_coord_taps: bool = True,
        high_risk_markers: Iterable[str] = (),
        forbidden_markers: Iterable[str] = (),
        confirm_fn: Optional[Callable[[str], str]] = None,
    ) -> None:
        self.auto_confirm_low = auto_confirm_low
        self.auto_confirm_medium = auto_confirm_medium
        self.auto_confirm_all = auto_confirm_all
        self.confirm_coord_taps = confirm_coord_taps
        self.high_risk_markers = tuple(high_risk_markers)
        self.forbidden_markers = tuple(forbidden_markers)
        self._high_norm = [normalize_text(m) for m in self.high_risk_markers if normalize_text(m)]
        self._forbidden_norm = [normalize_text(m) for m in self.forbidden_markers if normalize_text(m)]
        self.confirm_fn = confirm_fn or input

    def is_forbidden(self, target_text: str) -> bool:
        t = normalize_text(target_text)
        return any(f in t for f in self._forbidden_norm)

    def classify(self, action: Action, target_text: str = "") -> RiskLevel:
        level = action.risk  # LLM 自评作为基线
        t = normalize_text(target_text)
        # 黑名单与高风险标志词都升级到 HIGH（黑名单本应在 loop 里直接拦，这里作纵深防御）
        if any(f in t for f in self._forbidden_norm) or any(h in t for h in self._high_norm):
            level = RiskLevel.HIGH
        if action.kind == "text" and level.rank < RiskLevel.MEDIUM.rank:
            level = RiskLevel.MEDIUM  # 文本输入有副作用，至少 medium
        return level

    def _needs_confirm(self, action: Action, level: RiskLevel, *, is_coordinate: bool) -> bool:
        if self.auto_confirm_all:
            return False
        if level == RiskLevel.HIGH:
            return True
        if level == RiskLevel.MEDIUM:
            return not self.auto_confirm_medium  # 文本输入等中风险：可配置自动放行（送出仍属 HIGH，照样确认）
        # 坐标/视觉点击无文本交叉校验，属不可验证动作，默认需确认
        if is_coordinate and self.confirm_coord_taps and action.kind in {"tap", "long_press"}:
            return True
        # 其余为 LOW 的元素类动作
        return not self.auto_confirm_low

    def approve(self, action: Action, desc: str, *, is_coordinate: bool = False) -> bool:
        """是否放行该动作。需确认时调用 confirm_fn；返回 False 表示拒绝（调用方应转人工接管）。"""
        # 始终用 desc 参与判级（即便坐标模式 desc 多为「坐标(x,y)」无标志词，也保留 action.risk 基线）；
        # is_coordinate 只影响 _needs_confirm（坐标点击无法文本交叉校验，需额外确认）。
        level = self.classify(action, desc)
        if not self._needs_confirm(action, level, is_coordinate=is_coordinate):
            return True
        prompt = (
            f"\n[需要确认] 风险={level.value} 动作={action.kind} 目标={desc!r} "
            f"原因={action.reason!r}\n是否执行？(y/N) "
        )
        try:
            answer = self.confirm_fn(prompt)
        except (EOFError, KeyboardInterrupt):
            answer = ""
        ok = str(answer).strip().lower() in _YES
        log.info("人工确认：动作=%s 目标=%r 风险=%s -> %s", action.kind, desc, level.value, "放行" if ok else "拒绝")
        return ok
