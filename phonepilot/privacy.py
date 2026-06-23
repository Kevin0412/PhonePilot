"""隐私保护：敏感 app 拦截 + 屏幕内容脱敏。

agent 能看到用户看到的一切——相册、备忘录、银行 app、聊天记录。
此模块在发给云端 LLM 之前拦截/脱敏，防止隐私泄露。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

# 默认敏感 app（包名或关键词），agent 不得进入这些 app
_DEF_BLOCKED_PACKAGES = (
    # 金融/支付
    "com.eg.android.AlipayGphone",  # 支付宝
    "com.tencent.mm.plugin.wallet",  # 微信支付（子页面）
    # 银行
    "com.icbc", "com.ccb", "com.boc", "com.cmb", "com.abchina",
    "com.chinamworld.bocmbci", "com.cmbchina",
    # 密码管理
    "com.android.providers.contacts",  # 通讯录数据库
)

# 包名包含这些关键词的也视为敏感
_DEF_BLOCKED_KEYWORDS = (
    "bank", "pay", "wallet", "password", "vault", "authenticator", "otp",
)

# 默认敏感 Activity 关键词（即使 app 不在黑名单，这些页面也拦截）
_DEF_BLOCKED_ACTIVITIES = (
    "PayActivity", "TransferActivity", "BankCard", "PasswordInput",
    "WalletActivity", "PaymentActivity",
)

# 屏幕内容脱敏：匹配到这些模式的 OCR 文本会被替换为 [已脱敏]
_DEF_REDACT_PATTERNS = (
    r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",  # 银行卡号
    r"\b\d{3}[-.]?\d{4}[-.]?\d{4}\b",                 # 手机号
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",  # 邮箱
    r"(?:密码|password|pwd|PIN)[\s:：]*\S+",           # 密码字段
    r"(?:余额|balance|可用)[\s:：]*[¥￥$]?\s*[\d,.]+",  # 余额
    r"(?:身份证|ID)[\s:：]*\d{6,}",                    # 身份证号
    r"(?:验证码|OTP|code)[\s:：]*\d{4,}",              # 验证码
)


@dataclass
class PrivacyGuard:
    """隐私守卫：拦截敏感 app + 脱敏屏幕内容。"""

    blocked_packages: tuple[str, ...] = _DEF_BLOCKED_PACKAGES
    blocked_keywords: tuple[str, ...] = _DEF_BLOCKED_KEYWORDS
    blocked_activities: tuple[str, ...] = _DEF_BLOCKED_ACTIVITIES
    redact_patterns: tuple[str, ...] = _DEF_REDACT_PATTERNS
    enabled: bool = True
    # 用户可临时授权某些 app
    allowed_overrides: set[str] = field(default_factory=set)

    def is_app_blocked(self, package: str) -> bool:
        """检查 app 包名是否被拦截。"""
        if not self.enabled or not package:
            return False
        if package in self.allowed_overrides:
            return False
        if package in self.blocked_packages:
            return True
        pkg_lower = package.lower()
        return any(kw in pkg_lower for kw in self.blocked_keywords)

    def is_activity_blocked(self, foreground: str) -> bool:
        """检查当前 Activity 是否是敏感页面。"""
        if not self.enabled or not foreground:
            return False
        pkg = foreground.split("/", 1)[0]
        if pkg in self.allowed_overrides:
            return False
        if self.is_app_blocked(pkg):
            return True
        return any(kw in foreground for kw in self.blocked_activities)

    def redact_text(self, text: str) -> str:
        """脱敏文本中的敏感信息。"""
        if not self.enabled or not text:
            return text
        result = text
        for pat in self.redact_patterns:
            result = re.sub(pat, "[已脱敏]", result, flags=re.IGNORECASE)
        return result

    def redact_image(self, image: Any, elements: list) -> Any:
        """对截图中的敏感区域打马赛克（基于 OCR 元素的文字匹配）。"""
        if not self.enabled or image is None:
            return image
        import cv2
        import numpy as np

        redacted = image.copy()
        for e in elements:
            if not e.text:
                continue
            if self.redact_text(e.text) != e.text:
                # 这个元素包含敏感信息，对其包围盒区域打马赛克
                x1, y1, x2, y2 = e.bounds
                if x2 > x1 and y2 > y1:
                    roi = redacted[y1:y2, x1:x2]
                    if roi.size > 0:
                        # 缩小再放大 = 马赛克效果
                        small = cv2.resize(roi, (max(1, (x2 - x1) // 8), max(1, (y2 - y1) // 8)))
                        redacted[y1:y2, x1:x2] = cv2.resize(small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
        return redacted

    def redact_elements(self, elements: list) -> list:
        """脱敏元素列表中的敏感文字（就地修改 text 字段）。"""
        if not self.enabled:
            return elements
        from .types import ScreenElement
        out = []
        for e in elements:
            redacted = self.redact_text(e.text)
            if redacted != e.text:
                out.append(ScreenElement(
                    text=redacted, score=e.score, box=e.box,
                    style=e.style, kind=e.kind,
                ))
            else:
                out.append(e)
        return out

    def check_launch(self, package: str) -> Optional[str]:
        """launch 前检查，返回拦截原因或 None（允许）。"""
        if self.is_app_blocked(package):
            return f"隐私保护：{package} 是敏感应用（金融/密码类），禁止 agent 访问"
        return None

    def check_foreground(self, foreground: str) -> Optional[str]:
        """每帧检查前台，返回拦截原因或 None（允许）。"""
        if self.is_activity_blocked(foreground):
            return f"隐私保护：当前页面 {foreground} 包含敏感内容，agent 自动退出"
        return None

    @classmethod
    def from_config(cls, cfg: dict) -> "PrivacyGuard":
        pc = cfg.get("privacy", {})
        # config 里的列表是追加到默认值，不是替换（空列表不覆盖默认黑名单）
        extra_pkgs = tuple(pc.get("blocked_packages") or ())
        extra_kws = tuple(pc.get("blocked_keywords") or ())
        return cls(
            enabled=pc.get("enabled", True),
            blocked_packages=_DEF_BLOCKED_PACKAGES + extra_pkgs,
            blocked_keywords=_DEF_BLOCKED_KEYWORDS + extra_kws,
            blocked_activities=tuple(pc.get("blocked_activities") or _DEF_BLOCKED_ACTIVITIES),
            redact_patterns=tuple(pc.get("redact_patterns") or _DEF_REDACT_PATTERNS),
        )
