from phonepilot.perception.signature import (
    contains,
    find_candidate,
    find_element,
    normalize_text,
    page_signature,
)
from tests.conftest import el


def test_normalize_strips_space_punct_and_lowercases():
    assert normalize_text(" 立即 购买！") == "立即购买"
    assert normalize_text("Wi-Fi") == "wifi"


def test_signature_stable_across_volatile_text_and_reorder():
    f1 = [el("12:30"), el("85%"), el("设置"), el("WLAN"), el("蓝牙")]
    f2 = [el("蓝牙"), el("13:05"), el("WLAN"), el("72%"), el("设置")]
    assert page_signature(f1) == page_signature(f2)


def test_signature_changes_on_semantic_change():
    a = [el("设置"), el("WLAN")]
    b = [el("设置"), el("可用网络")]
    assert page_signature(a) != page_signature(b)


def test_countdown_normalized():
    a = [el("距开售还剩 03:21"), el("立即购买")]
    b = [el("距开售还剩 00:05"), el("立即购买")]
    assert page_signature(a) == page_signature(b)


def test_contains_returns_marker():
    els = [el("蓝牙"), el("WLAN")]
    assert contains(els, ["不存在", "蓝牙"]) == "蓝牙"
    assert contains(els, ["不存在"]) is None


def test_find_element_exact_contains_fuzzy():
    els = [el("立即购买"), el("取消")]
    assert find_element(els, "立即购买")[0] == 0
    assert find_element(els, "购买")[0] == 0  # 包含
    assert find_element([el("立即抢购")], "立即抢買") is not None  # 模糊


def test_find_element_rejects_blacklisted():
    els = [el("已预填抢票信息")]
    assert find_element(els, "抢票", reject_markers=["预填抢票信息"]) is None


def test_find_candidate_priority_order():
    els = [el("WLAN"), el("蓝牙")]
    idx, _e, marker = find_candidate(els, ["不在", "WLAN"])
    assert marker == "WLAN" and idx == 0
