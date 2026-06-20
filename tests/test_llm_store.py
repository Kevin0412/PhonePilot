import threading
import time

from phonepilot.agent.llm_store import RuntimeLlmStore
from phonepilot.types import Action


def test_get_returns_fresh_result():
    s = RuntimeLlmStore("t", min_interval=0.0)
    assert s.start("k", lambda: Action(kind="tap", index=1, text="a"))
    time.sleep(0.05)
    got = s.get("k", max_age=5)
    assert got is not None and got.index == 1


def test_result_expires_by_max_age():
    s = RuntimeLlmStore("t", min_interval=0.0)
    s.start("k", lambda: Action(kind="tap", index=1, text="a"))
    time.sleep(0.05)
    assert s.get("k", max_age=0.0) is None  # 立即过期


def test_start_clears_stale_result_before_new_worker_ready():
    s = RuntimeLlmStore("t", min_interval=0.0)
    s.start("k", lambda: Action(kind="tap", index=1, text="old"))
    time.sleep(0.05)
    assert s.get("k", 5) is not None  # 有旧结果

    started, release = threading.Event(), threading.Event()

    def slow():
        started.set()
        release.wait(1.0)
        return Action(kind="tap", index=2, text="new")

    assert s.start("k", slow) is True
    started.wait(1.0)
    # 旧结果已被清除，新结果尚未就绪 -> None（不会返回陈旧值）
    assert s.get("k", 5) is None
    release.set()
    time.sleep(0.1)
    got = s.get("k", 5)
    assert got is not None and got.index == 2


def test_busy_slot_drops_new_request():
    s = RuntimeLlmStore("t", min_interval=0.0)
    release = threading.Event()

    def slow():
        release.wait(1.0)
        return Action(kind="wait")

    assert s.start("k", slow) is True
    # 槽位忙时第二个请求被丢弃
    assert s.start("k2", lambda: Action(kind="wait")) is False
    release.set()
    time.sleep(0.05)
