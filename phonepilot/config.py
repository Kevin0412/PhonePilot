"""配置加载：内置默认 + config.yaml + CLI 覆盖 + .env（密钥仅走环境变量）。

返回一个普通嵌套 dict；调用方用 ``dataclass_from`` 把 loop/budget 段构造成对应 dataclass。
密钥永不写进配置文件，只从环境变量读取（``_load_dotenv`` 会把 .env 注入 os.environ）。
"""

from __future__ import annotations

import copy
import dataclasses
import os
from pathlib import Path
from typing import Any, Optional

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


DEFAULTS: dict[str, Any] = {
    "executor": {"kind": "mock", "serial": None, "adb_path": "adb", "mock_frames_dir": "examples/mock_frames"},
    "ocr": {"kind": "rapidocr", "min_score": 0.55, "use_cuda": False, "scale": 1.0},
    "icons": {"enabled": True, "template_dir": "phonepilot/perception/assets/icons"},
    # UI 控件检测(YOLO)：model_path 文件存在就自动启用，不存在静默回退 CV。
    "ui_detect": {"model_path": "models/omniparser_onnx/onnx/model.onnx", "confidence": 0.15, "long_side": 1280},
    # 多源感知融合：把 OCR + CV画框/检测器 + 控件树 全部打包给 LLM（LLM 选 index，坐标来自包围盒，不靠 VL）。
    # ui=控件树(UIAutomator)：能看到截图拍不到的系统浮层(如华为主/分身选择器)、给精确 bounds 与 content-desc。
    "perception": {"cv": True, "ui": True},
    "llm": {
        "kind": "mock",
        "fast": {"thinking": False, "max_tokens": 160, "temperature": 0.0},
        "rich": {"thinking": True, "max_tokens": 512, "temperature": 0.0},
        "deepseek": {"model": "deepseek-v4-flash", "base_url": "https://api.deepseek.com", "timeout": 30.0, "env_key": None},
        "claude": {
            "model": "claude-sonnet-4-5",
            "base_url": "https://api.anthropic.com",
            "timeout": 30.0,
            "anthropic_version": "2023-06-01",
            "env_key": None,
        },
        # 通用 OpenAI 兼容（MiMo/Qwen/Kimi/GLM 文本）。base_url/model 留空则用 provider 预置。
        "openai": {
            "provider": "mimo",
            "base_url": None,
            "model": None,
            "env_key": None,
            "timeout": 30.0,
            "json_mode": False,
        },
    },
    "vision": {
        "enabled": False,
        "provider": "mimo",   # mimo-v2.5 实测可定位控件；qwen/glm 需自备 key
        "base_url": None,
        "model": None,
        "timeout": 30.0,
        "confirm_taps": True,
        "max_calls": 4,
    },
    "privacy": {
        "enabled": True,
        # 默认黑名单见 privacy.py；这里可追加
        "blocked_packages": [],
        "blocked_keywords": [],
    },
    "risk": {
        "auto_confirm_low": True,
        "auto_confirm_medium": False,
        "auto_confirm_all": False,
        "confirm_coord_taps": True,
        "high_risk_markers": ["支付", "付款", "删除", "卸载", "转账", "发送", "授权", "格式化", "确认支付", "立即支付", "恢复出厂"],
        "forbidden_markers": ["立即支付", "确认支付", "确认付款"],
    },
    "loop": {
        "poll_interval": 0.3,
        "after_click_delay": 0.4,
        "transition_timeout": 12.0,
        "transition_threshold": 0.025,
        "llm_max_age": 4.0,
        "vision_max_age": 6.0,
        "llm_min_interval": 3.0,
        "foreground_interval": 2.0,
    },
    "budget": {
        "max_steps": 30,
        "max_recover_calls": 6,
        "max_rich_calls": 8,
        "max_vision_calls": 4,
        "max_no_change_retries": 2,
        "max_scrolls": 12,
        "max_replans": 3,
        "unknown_stop_seconds": 8.0,
        "loading_stop_seconds": 45.0,
        "total_timeout_s": 600.0,
    },
    "run_dir": "./runs",
}


def _deep_merge(base: dict, over: Optional[dict]) -> dict:
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def _load_dotenv() -> None:
    """把 .env 注入 os.environ（手动解析，已存在的环境变量不覆盖）。

    不走 python-dotenv：部分版本会把行内注释当成值（见 .env.example 的坑），这里自己解析更可控。
    """
    env_path = Path(".env")
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        # 去掉行内注释（空格+#…），避免把示例注释当成 key 值
        val = val.split(" #", 1)[0].strip().strip("'\"")
        if val.startswith("#"):
            val = ""
        if key and val and key not in os.environ:
            os.environ[key] = val


def load_config(path: Optional[str] = None, overrides: Optional[dict] = None) -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    candidate = Path(path) if path else Path("config.yaml")
    if candidate.exists():
        if yaml is None:
            raise RuntimeError("读取 YAML 配置需要 pyyaml")
        data = yaml.safe_load(candidate.read_text(encoding="utf-8")) or {}
        _deep_merge(cfg, data)
    _deep_merge(cfg, overrides)
    _load_dotenv()
    return cfg


def dataclass_from(cls, data: Optional[dict]):
    """用 dict 构造 dataclass，自动忽略多余/缺失键（缺失用 dataclass 默认）。"""
    names = {f.name for f in dataclasses.fields(cls)}
    return cls(**{k: v for k, v in (data or {}).items() if k in names})
