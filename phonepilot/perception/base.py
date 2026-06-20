"""感知层抽象接口：OCR 引擎与图标匹配器。

图像以 ``numpy.ndarray``（BGR, HxWx3）传递，但此处类型标注用 ``Any`` 以免接口模块
强依赖 numpy —— 真正触碰像素的是各适配器，且仅在其内部懒加载 cv2/onnxruntime。
"""

from __future__ import annotations

import abc
from typing import Any

from ..types import ScreenElement


class OcrEngine(abc.ABC):
    """把一帧图像识别为结构化屏幕元素。"""

    @abc.abstractmethod
    def read(self, image: Any, *, scale: float = 1.0) -> list[ScreenElement]:
        """返回该帧的文本元素列表（含坐标 box 与样式 style）。"""
        raise NotImplementedError


class IconMatcher(abc.ABC):
    """识别无文字的常见图标（返回/菜单/搜索/关闭/更多…）。

    返回的元素 ``kind="icon"``、``text="[icon:<name>]"``，与 OCR 文本元素一起并入
    Observation，从而让本地规则、LLM 与签名都能引用图标。
    """

    @abc.abstractmethod
    def match(self, image: Any) -> list[ScreenElement]:
        raise NotImplementedError


class NoopIconMatcher(IconMatcher):
    """空实现：不识别任何图标。MVP 在未配置模板时的默认。"""

    def match(self, image: Any) -> list[ScreenElement]:  # noqa: D401
        return []
