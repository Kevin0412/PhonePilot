"""平台 Profile：外部化的标志词与动作规则（移植参考项目的 JSON profile 思路）。

把「加载/恢复/硬停止/黑名单」等文字规则放进 JSON，按 App 适配，零代码扩展。纯数据，
无平台依赖。``hard_stop_markers`` 命中即交人工；``forbidden_markers`` 永不自动点击。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

# 通用默认：覆盖常见危险/加载/恢复文字，适配新 App 时再细化。
_DEF_FORBIDDEN = ("立即支付", "确认支付", "确认付款", "输入支付密码")
# 硬停标志走「整屏 OCR 子串匹配」，泛用 agent 会浏览任意内容(朋友圈/聊天/文章)，
# 故只保留几乎不会出现在正常内容里的「阻断性安全/付款页」短语，避免内容词误触发硬停。
# 通用内容易误报的词(验证码/付款/成功/售罄/暂无余票)一律不进硬停——
# 真遇验证码/支付页由 ACT prompt 让 LLM 返回 stop、点击层再由 forbidden_markers+风险闸门兜底。
_DEF_PAYMENT = ("确认支付", "立即支付", "输入支付密码", "请输入支付密码")
_DEF_MANUAL = ("拖动滑块", "滑动验证", "向右滑动验证", "完成安全验证", "人机验证")
_DEF_SUCCESS: tuple[str, ...] = ()
_DEF_LOADING = ("加载中", "正在加载", "请稍候", "排队中", "正在搜索", "玩命加载")
_DEF_RECOVERY = ("网络异常", "网络错误", "加载失败", "出错了", "系统繁忙", "前方拥挤")
_DEF_RECOVERY_ACTIONS = ("重试", "再试一次", "重新加载", "刷新", "知道了", "我知道了", "确定")


@dataclass
class Profile:
    name: str = "generic"
    expected_package: str = ""
    forbidden_markers: tuple[str, ...] = _DEF_FORBIDDEN
    payment_stop_markers: tuple[str, ...] = _DEF_PAYMENT
    manual_stop_markers: tuple[str, ...] = _DEF_MANUAL
    success_stop_markers: tuple[str, ...] = _DEF_SUCCESS
    loading_markers: tuple[str, ...] = _DEF_LOADING
    recovery_markers: tuple[str, ...] = _DEF_RECOVERY
    recovery_actions: tuple[str, ...] = _DEF_RECOVERY_ACTIONS

    @property
    def hard_stop_markers(self) -> tuple[str, ...]:
        """命中即交人工接管的页面级标志（支付/验证码/成功）。"""
        return self.payment_stop_markers + self.manual_stop_markers + self.success_stop_markers

    @classmethod
    def from_dict(cls, data: dict) -> "Profile":
        def pick(key: str, default: tuple[str, ...]) -> tuple[str, ...]:
            v = data.get(key)
            return tuple(str(x) for x in v) if isinstance(v, list) else default

        return cls(
            name=str(data.get("name", "generic")),
            expected_package=str(data.get("expected_package", "")),
            forbidden_markers=pick("forbidden_markers", _DEF_FORBIDDEN),
            payment_stop_markers=pick("payment_stop_markers", _DEF_PAYMENT),
            manual_stop_markers=pick("manual_stop_markers", _DEF_MANUAL),
            success_stop_markers=pick("success_stop_markers", _DEF_SUCCESS),
            loading_markers=pick("loading_markers", _DEF_LOADING),
            recovery_markers=pick("recovery_markers", _DEF_RECOVERY),
            recovery_actions=pick("recovery_actions", _DEF_RECOVERY_ACTIONS),
        )

    @classmethod
    def load(cls, path: str | Path | None) -> "Profile":
        if not path:
            return cls()
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"profile 文件不存在：{path}")
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"profile JSON 解析失败 {path}：{exc}") from exc
        return cls.from_dict(data)
