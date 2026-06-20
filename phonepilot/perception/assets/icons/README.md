# 图标模板目录

放置常见图标的灰度 PNG，文件名即图标名（`back.png`、`menu.png`、`search.png`、`close.png`、`more.png`、`share.png` …）。

`TemplateIconMatcher`（`phonepilot/perception/icons.py`）会用 `cv2.matchTemplate` 在截图中定位，命中后产出 `[icon:<name>]` 元素，与 OCR 文本一并进入决策。

启用：`config.yaml` 里 `icons.enabled: true`。未放模板时图标匹配自动跳过；未知图标交由视觉 LLM 兜底。
