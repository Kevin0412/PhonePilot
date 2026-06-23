"""隐私保护模块测试。"""

from phonepilot.privacy import PrivacyGuard


def test_blocked_package():
    g = PrivacyGuard()
    assert g.is_app_blocked("com.eg.android.AlipayGphone")
    assert not g.is_app_blocked("com.tencent.mm")


def test_blocked_keyword():
    g = PrivacyGuard()
    assert g.is_app_blocked("com.icbc.mobilebanking")
    assert g.is_app_blocked("com.some.wallet.app")


def test_allowed_override():
    g = PrivacyGuard()
    assert g.is_app_blocked("com.eg.android.AlipayGphone")
    g.allowed_overrides.add("com.eg.android.AlipayGphone")
    assert not g.is_app_blocked("com.eg.android.AlipayGphone")


def test_activity_blocked():
    g = PrivacyGuard()
    assert g.is_activity_blocked("com.eg.android.AlipayGphone/com.alipay.mobile.MainActivity")
    assert g.is_activity_blocked("com.tencent.mm/com.tencent.mm.plugin.wallet.PayActivity")
    assert not g.is_activity_blocked("com.tencent.mm/com.tencent.mm.ui.LauncherUI")


def test_redact_phone():
    g = PrivacyGuard()
    assert "[已脱敏]" in g.redact_text("手机号 138-1234-5678 请联系")


def test_redact_bank_card():
    g = PrivacyGuard()
    assert "[已脱敏]" in g.redact_text("卡号 6222 0200 1234 5678")


def test_redact_password():
    g = PrivacyGuard()
    assert "[已脱敏]" in g.redact_text("密码: abc123xyz")


def test_redact_balance():
    g = PrivacyGuard()
    assert "[已脱敏]" in g.redact_text("余额：¥12,345.67")


def test_disabled():
    g = PrivacyGuard(enabled=False)
    assert not g.is_app_blocked("com.eg.android.AlipayGphone")
    assert g.redact_text("密码: secret") == "密码: secret"


def test_check_launch():
    g = PrivacyGuard()
    assert g.check_launch("com.eg.android.AlipayGphone") is not None
    assert g.check_launch("tv.danmaku.bili") is None


def test_from_config_default():
    g = PrivacyGuard.from_config({})
    assert g.enabled
    assert g.is_app_blocked("com.eg.android.AlipayGphone")
