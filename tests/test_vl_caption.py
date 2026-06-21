"""VL 标注的标签提取逻辑测试（纯逻辑，不调 API）。"""

from phonepilot.perception.vl_caption import _extract_labels


def test_extract_json_labels():
    text = '{"0":"搜索","1":"返回","2":"添加"}'
    labels = _extract_labels(text)
    assert labels == {0: "搜索", 1: "返回", 2: "添加"}


def test_extract_json_from_mixed_text():
    text = '这是分析结果：\n{"0":"搜索","3":"菜单"}\n以上是控件。'
    labels = _extract_labels(text)
    assert labels == {0: "搜索", 3: "菜单"}


def test_extract_from_reasoning_text():
    text = '编号0的框是搜索图标，控件名"搜索"；编号3是菜单按钮，控件名"菜单"。'
    labels = _extract_labels(text)
    assert 0 in labels and labels[0] == "搜索"
    assert 3 in labels and labels[3] == "菜单"


def test_extract_colon_format():
    text = '0: "搜索"\n1: "返回"\n2: "添加"'
    labels = _extract_labels(text)
    assert labels[0] == "搜索"
    assert labels[1] == "返回"


def test_extract_empty_text():
    assert _extract_labels("") == {}
    assert _extract_labels("没有任何标签信息") == {}
