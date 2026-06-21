"""执行层抽象：类 ADB 执行能力的统一接口。

设计要点：动作原语极简（截图 + 注入事件），高层手势在 ``agent/gestures.py`` 用 swipe
组合实现。所有具体执行器（ADB / 原生 App 的 Shizuku / Mock）跑的是**同一组 shell 命令串**
（``input tap`` / ``screencap`` …），因此从 PC 的 ADB 迁移到设备上的 Shizuku 几乎零成本。
"""

from __future__ import annotations

import abc
from typing import Any


class Executor(abc.ABC):
    """统一执行接口。``screencap`` 返回 BGR ``numpy.ndarray``。"""

    @abc.abstractmethod
    def screencap(self) -> Any:
        """截图，返回 BGR ndarray (HxWx3)。"""
        raise NotImplementedError

    @abc.abstractmethod
    def tap(self, x: int, y: int) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def swipe(self, x1: int, y1: int, x2: int, y2: int, ms: int = 300) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def text(self, value: str) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def key(self, keycode: str) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def foreground(self) -> str:
        """当前前台 window/包名（best-effort，失败返回空串）。"""
        raise NotImplementedError

    # ---- 基于上面原语的默认实现（子类可覆盖）---- #
    def long_press(self, x: int, y: int, ms: int = 700) -> None:
        """长按 = 原地 swipe 一段时长。"""
        self.swipe(x, y, x, y, ms)

    def back(self) -> None:
        self.key("KEYCODE_BACK")

    def home(self) -> None:
        self.key("KEYCODE_HOME")

    def recent_apps(self) -> None:
        self.key("KEYCODE_APP_SWITCH")

    def launch_app(self, package: str) -> None:
        """用命令直接启动 app（能靠指令就不点图标）。子类实现（ADB: monkey/am；App: Shizuku 同命令）。"""
        raise NotImplementedError("此执行器不支持 launch_app")

    def start_intent(self, action: str, data: str = "") -> None:
        """用命令直达某页面（如系统设置子页）。子类实现（ADB: am start -a <action>）。"""
        raise NotImplementedError("此执行器不支持 start_intent")

    def keep_awake(self) -> None:
        """尽量保持屏幕常亮（可选，子类按平台实现）。"""

    def close(self) -> None:
        """释放资源（可选）。"""
