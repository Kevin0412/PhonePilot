"""LLM 客户端单测：monkeypatch http_post_json，不触网，只验证请求构造与解析。"""

import pytest

import phonepilot.planner.llm_openai as om
from phonepilot.planner.llm_openai import OpenAICompatClient


def _reply(content):
    return {"choices": [{"message": {"content": content}}]}


def test_openai_compat_payload_and_parse(monkeypatch):
    captured = {}

    def fake_post(base_url, path, headers, payload, timeout, retries=2):
        captured.update(base_url=base_url, path=path, headers=headers, payload=payload)
        return _reply('{"action":"tap","index":1,"text":"WLAN"}')

    monkeypatch.setattr(om, "http_post_json", fake_post)
    c = OpenAICompatClient(provider="mimo", api_key="k")
    out = c.complete_json("sys", "user", max_tokens=99)

    assert out == {"action": "tap", "index": 1, "text": "WLAN"}
    assert captured["base_url"] == "https://token-plan-cn.xiaomimimo.com/v1"
    assert captured["path"] == "/chat/completions"
    assert captured["payload"]["model"] == "mimo-v2.5"
    assert captured["payload"]["max_tokens"] == 99
    assert captured["headers"]["Authorization"] == "Bearer k"
    assert "response_format" not in captured["payload"]  # json_mode 默认 False


def test_openai_compat_json_mode_adds_response_format(monkeypatch):
    captured = {}

    def fake_post(base_url, path, headers, payload, timeout, retries=2):
        captured.update(payload=payload)
        return _reply('{"a":1}')

    monkeypatch.setattr(om, "http_post_json", fake_post)
    c = OpenAICompatClient(provider="mimo", api_key="k", json_mode=True)
    assert c.complete_json("s", "u") == {"a": 1}
    assert captured["payload"]["response_format"] == {"type": "json_object"}


def test_openai_compat_content_blocks(monkeypatch):
    monkeypatch.setattr(om, "http_post_json", lambda *a, **k: _reply([{"text": '{"x":'}, {"text": "2}"}]))
    c = OpenAICompatClient(provider="mimo", api_key="k")
    assert c.complete_json("s", "u") == {"x": 2}


def test_openai_compat_noisy_output_extracted(monkeypatch):
    monkeypatch.setattr(om, "http_post_json", lambda *a, **k: _reply('解析结果：{"action":"wait"} 完毕'))
    c = OpenAICompatClient(provider="mimo", api_key="k")
    assert c.complete_json("s", "u") == {"action": "wait"}


def test_openai_compat_missing_key_raises(monkeypatch):
    monkeypatch.delenv("MIMO_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        OpenAICompatClient(provider="mimo")


def test_explicit_base_url_model_override():
    c = OpenAICompatClient(provider="unknown", base_url="https://x/v1", model="m", api_key="k")
    assert c.base_url == "https://x/v1" and c.model == "m"
