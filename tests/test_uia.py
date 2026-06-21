from phonepilot.perception import uia


def test_input_center_finds_focused_edittext():
    xml = (
        '<hierarchy>'
        '<node class="android.widget.EditText" focused="false" bounds="[0,100][200,160]"/>'
        '<node class="android.widget.EditText" focused="true" bounds="[36,2535][1015,2629]"/>'
        '</hierarchy>'
    )
    # 优先 focused 的那个
    assert uia.input_center(xml) == ((36 + 1015) // 2, (2535 + 2629) // 2)


def test_no_edittext_returns_none():
    xml = '<node class="android.widget.TextView" text="王信" bounds="[100,500][300,560]"/>'
    assert uia.input_center(xml) is None


def test_empty_xml_returns_none():
    assert uia.input_center("") is None


def test_parse_controls_edit_and_clickable():
    xml = (
        '<node class="android.widget.EditText" bounds="[0,0][100,50]"/>'
        '<node class="android.widget.Button" clickable="true" text="发送" bounds="[200,0][300,50]"/>'
        '<node class="android.widget.ImageView" clickable="true" content-desc="删除" bounds="[400,0][450,50]"/>'
        '<node class="android.widget.TextView" text="不可点" clickable="false" bounds="[0,200][100,250]"/>'
    )
    ctrls = uia.parse_controls(xml)
    pairs = {(c.kind, c.text) for c in ctrls}
    assert ("edit", "") in pairs
    assert ("control", "发送") in pairs
    assert ("control", "删除") in pairs           # content-desc 图标按钮
    assert all(c.text != "不可点" for c in ctrls)  # 不可点的纯文本不计入控件
