"""Agent 持久化记忆：跨 run 记住学到的经验。

存储在 .phonepilot/memory.json，纯 JSON，包含：
- app_packages: {app名/关键词: 包名} — 学到的包名映射
- app_tips: {包名: [提示]} — 某 app 的操作经验
- user_prefs: {key: value} — 用户偏好（如"默认单程"）
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_DEFAULT_PATH = Path(".phonepilot/memory.json")

# 内置的常见 app 包名映射（不需要学习）
_BUILTIN_APPS: dict[str, str] = {
    "微信": "com.tencent.mm",
    "QQ": "com.tencent.mobileqq",
    "bilibili": "tv.danmaku.bili",
    "B站": "tv.danmaku.bili",
    "抖音": "com.ss.android.ugc.aweme",
    "淘宝": "com.taobao.taobao",
    "支付宝": "com.eg.android.AlipayGphone",
    "设置": "com.android.settings",
    "高德": "com.autonavi.minimap",
    "高德地图": "com.autonavi.minimap",
    "美团": "com.sankuai.meituan",
    "京东": "com.jingdong.app.mall",
    "小红书": "com.xingin.xhs",
    "知乎": "com.zhihu.android",
    "携程": "ctrip.android.view",
    "携程旅行": "ctrip.android.view",
    "去哪儿": "com.Qunar",
    "百度地图": "com.baidu.BaiduMap",
    "12306": "com.MobileTicket",
    "铁路12306": "com.MobileTicket",
    "deepseek": "com.deepseek.chat",
    "拼多多": "com.xunmeng.pinduoduo",
    "网易云音乐": "com.netease.cloudmusic",
}


class AgentMemory:
    """Agent 持久化记忆。"""

    def __init__(self, path: Path = _DEFAULT_PATH) -> None:
        self.path = path
        self._data: dict[str, Any] = {"app_packages": {}, "app_tips": {}, "user_prefs": {}}
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            try:
                self._data = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception as exc:  # noqa: BLE001
                log.debug("记忆加载失败: %s", exc)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data, ensure_ascii=False, indent=2), encoding="utf-8")

    def resolve_package(self, app_name: str) -> str | None:
        """根据 app 名/关键词查包名：先查内置→再查学到的→都没有返回 None。"""
        # 精确匹配
        if app_name in _BUILTIN_APPS:
            return _BUILTIN_APPS[app_name]
        learned = self._data.get("app_packages", {})
        if app_name in learned:
            return learned[app_name]
        # 模糊匹配（app名出现在键里）
        name_lower = app_name.lower()
        for k, v in _BUILTIN_APPS.items():
            if name_lower in k.lower() or k.lower() in name_lower:
                return v
        for k, v in learned.items():
            if name_lower in k.lower() or k.lower() in name_lower:
                return v
        return None

    def learn_package(self, app_name: str, package: str) -> None:
        """学到一个新的 app 名→包名映射。"""
        if app_name and package and app_name not in _BUILTIN_APPS:
            self._data.setdefault("app_packages", {})[app_name] = package
            self.save()
            log.info("记忆：学到 %s → %s", app_name, package)

    def add_tip(self, package: str, tip: str) -> None:
        """记录某 app 的操作经验/提示。"""
        tips = self._data.setdefault("app_tips", {}).setdefault(package, [])
        if tip not in tips:
            tips.append(tip)
            if len(tips) > 10:
                tips.pop(0)
            self.save()

    def get_tips(self, package: str) -> list[str]:
        return self._data.get("app_tips", {}).get(package, [])

    def set_pref(self, key: str, value: Any) -> None:
        self._data.setdefault("user_prefs", {})[key] = value
        self.save()

    def get_pref(self, key: str, default: Any = None) -> Any:
        return self._data.get("user_prefs", {}).get(key, default)

    def format_context(self) -> str:
        """格式化记忆为 LLM 可读的上下文字符串。"""
        parts: list[str] = []
        learned = self._data.get("app_packages", {})
        if learned:
            parts.append("学到的 app 包名：" + "、".join(f"{k}={v}" for k, v in learned.items()))
        prefs = self._data.get("user_prefs", {})
        if prefs:
            parts.append("用户偏好：" + "、".join(f"{k}={v}" for k, v in prefs.items()))
        return "\n".join(parts)
