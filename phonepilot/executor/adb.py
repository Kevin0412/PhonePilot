"""ADB 执行器（PC 开发期驱动真机）。

通过 `adb` 子进程执行 `input tap/swipe/text/keyevent` 与 `exec-out screencap -p`。
这些命令串与原生 App 阶段的 Shizuku 完全一致 —— 届时只需把「subprocess 调 adb」
换成「Shizuku Binder 起进程」，命令体不变。cv2/numpy 仅在 screencap 时懒加载。
"""

from __future__ import annotations

import logging
import re
import subprocess
import time
from typing import Any, Optional

from .base import Executor

log = logging.getLogger(__name__)

_ADBKB_PKG = "com.android.adbkeyboard"
_ADBKB_IME = "com.android.adbkeyboard/.AdbIME"

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
        self._use_adbkb: Optional[bool] = None  # 懒检测 ADBKeyboard（中文输入）
        self._orig_ime: Optional[str] = None     # 切换前的输入法，close() 时恢复
        self._orig_show_ime: Optional[str] = None  # show_ime_with_hard_keyboard 原值

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

    def _has_adbkeyboard(self) -> bool:
        if self._use_adbkb is None:
            try:
                out = self._shell("pm", "list", "packages", _ADBKB_PKG, timeout=8)
                self._use_adbkb = _ADBKB_PKG in out
            except Exception:  # pragma: no cover
                self._use_adbkb = False
        return bool(self._use_adbkb)

    def _ensure_adbkb(self) -> None:
        """切到 ADBKeyboard（记住原输入法，close() 时恢复）。"""
        if self._orig_ime is None:
            cur = self._shell("settings", "get", "secure", "default_input_method", timeout=5).strip()
            self._orig_ime = cur or "__none__"
            # ADBKeyboard 可能被系统/厂商置为 disabled（enabled=0），先启用包，否则 ime/广播都无效
            self._shell("pm", "enable", _ADBKB_PKG, timeout=5)
            self._shell("ime", "enable", _ADBKB_IME, timeout=5)
            self._shell("ime", "set", _ADBKB_IME, timeout=5)
            # 外接物理键盘时系统默认不弹软输入法 → IME 不接管、广播无效。强制显示软 IME。
            self._orig_show_ime = self._shell(
                "settings", "get", "secure", "show_ime_with_hard_keyboard", timeout=5
            ).strip()
            self._shell("settings", "put", "secure", "show_ime_with_hard_keyboard", "1", timeout=5)
            time.sleep(0.6)

    def prepare_input(self) -> None:
        # 在聚焦输入框之前先切到 ADBKeyboard，确保聚焦时可编程输入法已就绪（顺序很关键）。
        if self._has_adbkeyboard():
            self._ensure_adbkb()

    def text(self, value: str) -> None:
        if not value:
            return
        # ADB 的 input text 只认 ASCII；中文用 ADBKeyboard 广播输入。
        if self._has_adbkeyboard():
            self._ensure_adbkb()
            msg = value.replace("\\", "\\\\").replace("'", "'\\''")
            self._run("shell", f"am broadcast -a ADB_INPUT_TEXT --es msg '{msg}'", timeout=8)
        else:
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

    def _resolve_launch_activity(self, package: str) -> str:
        """解析 app 主启动 Activity（cmd package resolve-activity --brief）。失败返回空。

        输出末行形如 ``com.tencent.mm/.ui.LauncherUI``。
        """
        try:
            out = self._shell("cmd", "package", "resolve-activity", "--brief", package, timeout=8)
        except Exception as exc:  # noqa: BLE001
            log.debug("resolve-activity %s 失败：%s", package, exc)
            return ""
        for line in reversed(out.splitlines()):
            line = line.strip()
            if line.startswith(package + "/"):
                return line
        return ""

    def launch_app(self, package: str) -> None:
        # 优先 am start 到解析出的主 Activity：monkey 对开了「应用分身」的 app 会静默失灵
        # （事件已注入但不到前台），唯有 am start 能正常触发华为主/分身选择器。解析失败再回退 monkey。
        activity = self._resolve_launch_activity(package)
        launched = False
        if activity:
            try:
                out = self._shell("am", "start", "-n", activity, timeout=10)
                launched = "Error" not in out and "Exception" not in out
            except Exception as exc:  # noqa: BLE001
                log.debug("am start %s 失败：%s", activity, exc)
        if not launched:
            out = self._shell(
                "monkey", "-p", package, "-c", "android.intent.category.LAUNCHER", "1", timeout=10
            )
            if "No activities found" in out:
                raise RuntimeError(f"启动 {package} 失败：{out.strip()[:120]}")
        # 轮询前台直到目标包真的到前台（最多 ~6s）：app 可能慢启动；华为「应用分身」会先弹出
        # 主/分身选择器(HwChooser，截图拍不到)，需在此期间选主应用后 app 才到前台。固定 sleep 会两头落空。
        fg = ""
        handled_chooser = False
        for _ in range(12):
            time.sleep(0.5)
            fg = self.foreground()
            if any(k in fg for k in ("HwChooser", "ResolverActivity", "ChooserActivity")):
                if not handled_chooser:
                    self._handle_twin_chooser()
                    handled_chooser = True
                continue
            if fg.split("/", 1)[0] == package or package in fg:
                return  # 目标 app 已到前台
        raise RuntimeError(f"launch {package} 后未到前台（当前={fg or '未知'}），可能包名错误")

    def _handle_twin_chooser(self) -> None:
        """华为「应用分身」会弹出主/分身选择器(HwChooserActivity，截图拍不到但控件树可见)。
        自动选择主应用(content-desc 不含「分身」的那项)。"""
        try:
            fg = self.foreground()
        except Exception:  # pragma: no cover
            return
        if not any(k in fg for k in ("HwChooser", "ResolverActivity", "ChooserActivity")):
            return
        xml = self.ui_dump()
        if not xml:
            return
        for m in re.finditer(r"<node\b([^>]*?)/?>", xml):
            attrs = m.group(1)
            if "text1" not in attrs:  # 选择器的应用名条目 resource-id=android:id/text1
                continue
            desc = re.search(r'content-desc="([^"]*)"', attrs)
            bounds = re.search(r'bounds="\[(-?\d+),(-?\d+)\]\[(-?\d+),(-?\d+)\]"', attrs)
            if not (desc and bounds and desc.group(1)):
                continue
            if "分身" in desc.group(1) or "clone" in desc.group(1).lower():
                continue
            x1, y1, x2, y2 = (int(v) for v in bounds.groups())
            self.tap((x1 + x2) // 2, (y1 + y2) // 2)  # 选主应用
            time.sleep(2.0)
            return

    def start_intent(self, action: str, data: str = "") -> None:
        args = ["am", "start", "-a", action]
        if data:
            args += ["-d", data]
        out = self._shell(*args, timeout=10)
        if "Error" in out or "Exception" in out:
            raise RuntimeError(f"am start -a {action} 失败：{out.strip()[:160]}")

    def keep_awake(self) -> None:
        try:
            self._shell("svc", "power", "stayon", "usb", timeout=5)
        except Exception:  # pragma: no cover
            pass

    def ui_dump(self) -> str:
        try:
            self._shell("uiautomator", "dump", "/sdcard/pp_ui.xml", timeout=12)
            return self._shell("cat", "/sdcard/pp_ui.xml", timeout=8)
        except Exception as exc:  # noqa: BLE001 - 控件树拿不到就回退其它聚焦方式
            log.debug("uiautomator dump 失败：%s", exc)
            return ""

    def close(self) -> None:
        # 恢复用户原来的输入法（之前为中文输入切到了 ADBKeyboard）
        if self._orig_ime and self._orig_ime != "__none__":
            try:
                self._shell("ime", "set", self._orig_ime, timeout=5)
            except Exception:  # pragma: no cover
                pass
        if self._orig_show_ime is not None:
            restore = self._orig_show_ime if self._orig_show_ime in ("0", "1") else "0"
            try:
                self._shell("settings", "put", "secure", "show_ime_with_hard_keyboard", restore, timeout=5)
            except Exception:  # pragma: no cover
                pass
        self._orig_ime = None
        self._orig_show_ime = None
