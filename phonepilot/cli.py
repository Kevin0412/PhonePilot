"""命令行入口：run / inspect / replay。

组件全部按配置装配（执行器/OCR/图标/LLM/视觉/风险闸门/规划器），可被 CLI 标志覆盖。
无真机演示走 ``run --example examples/open_wlan.yaml``（自带任务+计划+帧+profile）。
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import sys
from pathlib import Path
from typing import Any, Optional

from .agent.budget import Budget
from .agent.loop import LoopConfig, run_agent
from .agent.observe import observe
from .config import dataclass_from, load_config
from .logging_setup import setup_logging
from .profile import Profile
from .risk.gate import RiskGate

log = logging.getLogger("phonepilot")


# --------------------------------------------------------------------------- #
# 组件装配
# --------------------------------------------------------------------------- #
def _load_mock_frames(spec: list[dict] | None, frames_dir: str | None):
    from .executor.mock import MockFrame

    if spec:
        return [MockFrame.from_dict(f) for f in spec]
    d = Path(frames_dir) if frames_dir else None
    if not d or not d.is_dir():
        raise RuntimeError(f"未找到 mock 帧目录：{frames_dir}（或用 --example 提供内联帧）")
    files = sorted(d.glob("*.json"))
    if not files:
        raise RuntimeError(f"{frames_dir} 下没有帧 JSON")
    return [MockFrame.from_dict(json.loads(f.read_text(encoding="utf-8"))) for f in files]


def build_executor(cfg: dict, *, mock_frames: list[dict] | None = None):
    kind = cfg["executor"]["kind"]
    if kind == "mock":
        from .executor.mock import MockExecutor

        return MockExecutor(_load_mock_frames(mock_frames, cfg["executor"].get("mock_frames_dir")))
    if kind == "adb":
        from .executor.adb import AdbExecutor

        return AdbExecutor(serial=cfg["executor"].get("serial"), adb_path=cfg["executor"].get("adb_path", "adb"))
    raise ValueError(f"未知执行器：{kind}")


def build_ocr(cfg: dict, executor):
    from .executor.mock import MockExecutor, MockOcrEngine

    if isinstance(executor, MockExecutor):
        return MockOcrEngine(executor)
    kind = cfg["ocr"]["kind"]
    if kind == "rapidocr":
        from .perception.rapidocr_engine import RapidOcrEngine

        return RapidOcrEngine(min_score=cfg["ocr"]["min_score"], use_cuda=cfg["ocr"]["use_cuda"])
    raise ValueError(f"未知 OCR 引擎：{kind}")


def build_icons(cfg: dict):
    if not cfg["icons"]["enabled"]:
        return None
    from .perception.icons import TemplateIconMatcher

    return TemplateIconMatcher(cfg["icons"]["template_dir"])


def build_llm(cfg: dict, *, mock_plan: dict | None = None):
    kind = cfg["llm"]["kind"]
    if kind == "mock":
        from .planner.llm_mock import MockLLMClient

        return MockLLMClient(plan=mock_plan)
    if kind == "deepseek":
        from .planner.llm_deepseek import DeepSeekClient

        ds = cfg["llm"]["deepseek"]
        return DeepSeekClient(model=ds["model"], base_url=ds["base_url"], timeout=ds["timeout"])
    if kind == "claude":
        from .planner.llm_claude import ClaudeClient

        c = cfg["llm"]["claude"]
        return ClaudeClient(
            model=c["model"], base_url=c["base_url"], timeout=c["timeout"],
            anthropic_version=c["anthropic_version"],
        )
    raise ValueError(f"未知 LLM：{kind}")


def build_vision(cfg: dict):
    if not cfg["vision"]["enabled"]:
        return None
    from .planner.llm_vision import OpenAICompatVisionClient

    v = cfg["vision"]
    return OpenAICompatVisionClient(provider=v["provider"], base_url=v["base_url"], model=v["model"], timeout=v["timeout"])


def build_gate(cfg: dict, confirm_fn=None) -> RiskGate:
    r = cfg["risk"]
    return RiskGate(
        auto_confirm_low=r["auto_confirm_low"],
        auto_confirm_all=r["auto_confirm_all"],
        confirm_coord_taps=r["confirm_coord_taps"],
        high_risk_markers=r["high_risk_markers"],
        forbidden_markers=r["forbidden_markers"],
        confirm_fn=confirm_fn,
    )


def build_planner(cfg: dict, llm, vision):
    from .planner.planner import Planner

    return Planner(llm, vision, fast=cfg["llm"]["fast"], rich=cfg["llm"]["rich"])


def _make_run_dir(cfg: dict, enabled: bool) -> Optional[Path]:
    if not enabled:
        return None
    base = Path(cfg.get("run_dir") or "./runs")
    d = base / _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    d.mkdir(parents=True, exist_ok=True)
    return d


# --------------------------------------------------------------------------- #
# 子命令
# --------------------------------------------------------------------------- #
def cmd_run(args, cfg: dict) -> int:
    mock_plan = None
    mock_frames = None
    profile_path = args.profile

    if args.example:
        try:
            bundle = _read_yaml(args.example)
        except Exception as exc:  # noqa: BLE001
            log.error("加载示例 bundle 失败（%s）：%s", args.example, exc)
            return 1
        cfg["executor"]["kind"] = "mock"
        cfg["llm"]["kind"] = "mock"
        task = bundle.get("task") or args.task
        mock_plan = bundle.get("mock_plan")
        mock_frames = bundle.get("mock_frames")
        profile_path = profile_path or bundle.get("profile")
    else:
        task = args.task

    if not task:
        log.error("缺少任务：用 --task \"...\" 或 --example <bundle.yaml>")
        return 1
    if cfg["llm"]["kind"] == "mock" and args.plan_file and mock_plan is None:
        try:
            pf = _read_yaml(args.plan_file)
        except Exception as exc:  # noqa: BLE001
            log.error("加载计划文件失败（%s）：%s", args.plan_file, exc)
            return 1
        mock_plan = pf.get("mock_plan") or pf

    try:
        profile = Profile.load(profile_path)
    except Exception as exc:  # noqa: BLE001
        log.error("加载 profile 失败：%s", exc)
        return 1

    executor = build_executor(cfg, mock_frames=mock_frames)
    ocr = build_ocr(cfg, executor)
    icons = build_icons(cfg)
    llm = build_llm(cfg, mock_plan=mock_plan)
    vision = build_vision(cfg)
    confirm_fn = (lambda _p: "y") if cfg["risk"]["auto_confirm_all"] else None
    gate = build_gate(cfg, confirm_fn)
    planner = build_planner(cfg, llm, vision)
    budget = dataclass_from(Budget, cfg["budget"])
    loop = dataclass_from(LoopConfig, cfg["loop"])
    run_dir = _make_run_dir(cfg, enabled=not args.no_run_dir)

    log.info("任务：%s", task)
    try:
        plan = planner.make_plan(task)
    except Exception as exc:  # noqa: BLE001
        log.error("规划失败：%s", exc)
        return 1

    rc = run_agent(
        plan, executor=executor, ocr=ocr, planner=planner, gate=gate, profile=profile,
        budget=budget, loop=loop, icons=icons, vision_enabled=vision is not None,
        run_dir=run_dir, dry_run=args.dry_run,
    )
    if run_dir:
        log.info("运行产物：%s", run_dir)
    return rc


def cmd_inspect(args, cfg: dict) -> int:
    mock_frames = None
    if args.example:
        try:
            bundle = _read_yaml(args.example)
        except Exception as exc:  # noqa: BLE001
            log.error("加载示例 bundle 失败（%s）：%s", args.example, exc)
            return 1
        cfg["executor"]["kind"] = "mock"
        mock_frames = bundle.get("mock_frames")
    executor = build_executor(cfg, mock_frames=mock_frames)
    ocr = build_ocr(cfg, executor)
    icons = build_icons(cfg)
    _img, obs = observe(executor, ocr, icons)
    print(f"前台：{obs.foreground or '(未知)'}")
    print(f"屏幕：{obs.image_w}x{obs.image_h}  元素 {len(obs.elements)} 个")
    print("-" * 60)
    for i, el in enumerate(obs.elements):
        print(f"[{i:2d}] {el.kind:4s} {el.style:14s} {el.center} {el.text!r}")
    print("-" * 60)
    print("页面签名：")
    print(obs.signature)
    return 0


def cmd_replay(args, cfg: dict) -> int:
    d = Path(args.runs_dir)
    files = sorted(d.glob("*.json"))
    if not files:
        log.error("%s 下没有审计 JSON", args.runs_dir)
        return 1
    for f in files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        note = data.get("note", {})
        print(f"\n=== {f.name} ===")
        print(f"前台：{data.get('foreground','')}  决策：{note}")
        print(f"签名：{(data.get('signature','') or '')[:200]}")
    return 0


# --------------------------------------------------------------------------- #
# argparse
# --------------------------------------------------------------------------- #
def _read_yaml(path: str) -> dict:
    import yaml

    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", help="配置文件路径（默认 ./config.yaml 若存在）")
    p.add_argument("--executor", choices=["mock", "adb"], help="覆盖执行器")
    p.add_argument("--serial", help="adb 设备序列号")
    p.add_argument("--adb-path", help="adb 可执行文件路径")
    p.add_argument("--ocr", choices=["rapidocr"], help="覆盖 OCR 引擎")
    p.add_argument("--profile", help="平台 profile JSON")
    p.add_argument("--example", help="无真机演示 bundle（含 task/计划/帧/profile）")
    p.add_argument("-v", "--verbose", action="store_true", help="DEBUG 日志")


def _overrides_from(args) -> dict:
    o: dict[str, Any] = {}
    if getattr(args, "executor", None):
        o.setdefault("executor", {})["kind"] = args.executor
    if getattr(args, "serial", None):
        o.setdefault("executor", {})["serial"] = args.serial
    if getattr(args, "adb_path", None):
        o.setdefault("executor", {})["adb_path"] = args.adb_path
    if getattr(args, "ocr", None):
        o.setdefault("ocr", {})["kind"] = args.ocr
    if getattr(args, "llm", None):
        o.setdefault("llm", {})["kind"] = args.llm
    if getattr(args, "vision", None) is not None:
        o.setdefault("vision", {})["enabled"] = args.vision
    if getattr(args, "auto_confirm_all", False):
        o.setdefault("risk", {})["auto_confirm_all"] = True
    if getattr(args, "run_dir", None):
        o["run_dir"] = args.run_dir
    return o


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="phonepilot", description="实验性 Android 手机 agent 框架")
    sub = parser.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("run", help="执行一个自然语言任务")
    _add_common(pr)
    pr.add_argument("--task", help="自然语言任务描述")
    pr.add_argument("--llm", choices=["mock", "deepseek", "claude"], help="覆盖 LLM")
    pr.add_argument("--plan-file", help="mock LLM 时的计划 YAML/JSON")
    pr.add_argument("--vision", dest="vision", action="store_true", default=None, help="启用视觉兜底")
    pr.add_argument("--no-vision", dest="vision", action="store_false", help="禁用视觉兜底")
    pr.add_argument("--dry-run", action="store_true", help="只预览当前屏幕的下一步决策，不执行（接真机前自检）")
    pr.add_argument("--auto-confirm-all", action="store_true", help="跳过所有人工确认（谨慎）")
    pr.add_argument("--run-dir", help="运行产物目录")
    pr.add_argument("--no-run-dir", action="store_true", help="不落盘运行产物")
    pr.set_defaults(func=cmd_run)

    pi = sub.add_parser("inspect", help="只感知一帧并打印元素/签名，不执行")
    _add_common(pi)
    pi.set_defaults(func=cmd_inspect)

    prp = sub.add_parser("replay", help="回放历史运行目录的审计 JSON")
    prp.add_argument("--runs-dir", required=True, help="某次 runs/<时间戳> 目录")
    prp.add_argument("-v", "--verbose", action="store_true")
    prp.set_defaults(func=cmd_replay)

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging(getattr(args, "verbose", False))
    if args.cmd == "replay":
        return args.func(args, {})
    cfg = load_config(getattr(args, "config", None), _overrides_from(args))
    return args.func(args, cfg)


if __name__ == "__main__":
    sys.exit(main())
