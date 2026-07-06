from phonepilot.perception.signature import (
    contains,
    find_candidate,
    find_element,
    normalize_text,
    page_signature,
)
from tests.conftest import el
from phonepilot.types import ScreenElement


def test_normalize_strips_space_punct_and_lowercases():
    assert normalize_text(" 立即 购买！") == "立即购买"
    assert normalize_text("Wi-Fi") == "wifi"
    assert normalize_text("WLAN+") == "wlan+"


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


def test_find_element_avoids_secondary_result_line_when_exact_matches_repeat():
    # 搜索结果常见布局：标题行「更多WLAN设置」下面有路径/分类行「WLAN」。
    # 精确匹配 WLAN 时不应优先点这个副标题，否则会打开相邻功能页。
    els = [
        ScreenElement.from_dict({"text": "更多WLAN设置", "box": [[250, 1000], [614, 1000], [614, 1055], [250, 1055]]}),
        ScreenElement.from_dict({"text": "WLAN", "box": [[248, 1078], [380, 1078], [380, 1124], [248, 1124]]}),
        ScreenElement.from_dict({"text": "WLAN", "box": [[248, 1848], [398, 1848], [398, 1904], [248, 1904]]}),
    ]
    hit = find_element(els, "WLAN")
    assert hit is not None
    assert hit[0] == 2


def test_find_element_short_ascii_token_does_not_match_longer_feature_name():
    els = [el("WLAN+"), el("WLAN直连"), el("公共WLAN推荐")]
    assert find_element(els, "WLAN") is None
