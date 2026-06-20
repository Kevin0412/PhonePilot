"""ADB 执行器（PC 开发期驱动真机）。

通过 `adb` 子进程执行 `input tap/swipe/text/keyevent` 与 `exec-out screencap -p`。
这些命令串与原生 App 阶段的 Shizuku 完全一致 —— 届时只需把「subprocess 调 adb」
换成「Shizuku Binder 起进程」，命令体不变。cv2/numpy 仅在 screencap 时懒加载。
"""

from __future__ import annotations

import logging
import re
import subprocess
from typing import Any, Optional

from .base import Executor

log = logging.getLogger(__name__)

_FOCUS_RE = re.compile(r"(?:mCurrentFocus|mFocusedApp|mResumedActivity)[^\n]*?([A-Za-z0-9_.]+/[A-Za-z0-9_.$]+)")


def _escape_text(value: str) -> str:
    """`input text` 转义：空格转 %s，转义 shell 特殊字符。"""
    out = value.replace("%", "%%").replace(" ", "%s")
    for ch in "()<>|;&*\\~\"'`$":
        out = out.replace(ch, "\\" + ch)
    return out


def _norm_keycode(code: str) -> str:
    code = str(code).strip()
    if code.isdigit():
        return code
    if not code.upper().startswith("KEYCODE_"):
        code = "KEYCODE_" + code.upper()
    return code


class AdbExecutor(Executor):
    def __init__(
        self,
        *,
        serial: Optional[str] = None,
        adb_path: str = "adb",
        timeout: float = 10.0,
    ) -> None:
        self.adb_path = adb_path
        self.serial = serial
        self.timeout = timeout
        self._base = [adb_path] + (["-s", serial] if serial else [])

    # ---- 底层 ---- #
    def _run(self, *args: str, timeout: Optional[float] = None) -> subprocess.CompletedProcess:
        return subprocess.run(
            self._base + list(args),
            capture_output=True,
            timeout=timeout or self.timeout,
        )

    def _shell(self, *args: str, timeout: Optional[float] = None) -> str:
        cp = self._run("shell", *args, timeout=timeout)
        return cp.stdout.decode("utf-8", errors="replace")

    def get_state(self) -> str:
        cp = self._run("get-state")
        return cp.stdout.decode("utf-8", errors="replace").strip()

    # ---- 接口实现 ---- #
    def screencap(self) -> Any:
        import cv2
        import numpy as np

        cp = self._run("exec-out", "screencap", "-p")
        data = cp.stdout
        if not data:
            raise RuntimeError("adb screencap 返回空数据；设备是否连接/授权？")
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError("screencap PNG 解码失败")
        return img

    def tap(self, x: int, y: int) -> None:
        self._shell("input", "tap", str(int(x)), str(int(y)), timeout=5)

    def swipe(self, x1: int, y1: int, x2: int, y2: int, ms: int = 300) -> None:
        self._shell(
            "input", "swipe", str(int(x1)), str(int(y1)), str(int(x2)), str(int(y2)),
            str(int(ms)), timeout=max(5.0, ms / 1000.0 + 3.0),
        )

    def text(self, value: str) -> None:
        if not value:
            return
        self._shell("input", "text", _escape_text(value), timeout=8)

    def key(self, keycode: str) -> None:
        self._shell("input", "keyevent", _norm_keycode(keycode), timeout=5)

    def foreground(self) -> str:
        try:
            out = self._shell("dumpsys", "window", timeout=8)
        except Exception as exc:  # pragma: no cover - 真机路径
            log.debug("foreground 查询失败: %s", exc)
            return ""
        m = _FOCUS_RE.search(out)
        return m.group(1) if m else ""

    def keep_awake(self) -> None:
        try:
            self._shell("svc", "power", "stayon", "usb", timeout=5)
        except Exception:  # pragma: no cover
            pass
