#!/usr/bin/env python3
"""AutoResearch 노드 CLI — 스윕 실행 / 리더보드 조회 / best export.

사용 예:
    python3 run.py run    --config config/run_example.yaml
    python3 run.py top     --leaderboard runs/leaderboard.jsonl -n 5
    python3 run.py export  --leaderboard runs/leaderboard.jsonl --out runs/export

torch / growing-memory-pytorch 가 없어도 mock 백엔드로 끝까지 동작한다(설계 T0).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# 패키지 import 경로 보정(스크립트 직접 실행 시)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from autoresearch.export import export_best
from autoresearch.leaderboard import Leaderboard
from autoresearch.loop import run_from_file


def cmd_run(args: argparse.Namespace) -> int:
    from autoresearch.config_io import load_run_config
    from autoresearch.loop import autoresearch_loop
    run_cfg = load_run_config(args.config)
    # CLI 오버라이드(배터리 실험용)
    if args.seed is not None:
        run_cfg["seed"] = args.seed
    if args.run_id:
        run_cfg["run_id"] = args.run_id
    if args.proxy:
        run_cfg.setdefault("proxy", {})["task"] = args.proxy
    if args.max_trials is not None:
        run_cfg.setdefault("search", {})["max_trials"] = args.max_trials
    if args.no_ssl:  # T0: SSL 고정(표현축 제거)
        run_cfg.get("space", {}).pop("ssl", None)
    summary = autoresearch_loop(run_cfg, args.leaderboard)
    print("\n=== summary ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def cmd_top(args: argparse.Namespace) -> int:
    lb = Leaderboard(args.leaderboard)
    rows = lb.top(args.n)
    print(json.dumps(lb.summary(), ensure_ascii=False, indent=2))
    print(f"\n=== top {args.n} ===")
    for i, r in enumerate(rows, 1):
        print(f"{i:>2}. {r.trial_id} proxy={r.proxy_score} full={r.full_score} "
              f"cost={r.cost} cfg={json.dumps(r.cfg, ensure_ascii=False)}")
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    result = export_best(args.leaderboard, args.out)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="AutoResearch 노드 CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("run", help="스윕 실행")
    pr.add_argument("--config", default="config/run_example.yaml")
    pr.add_argument("--leaderboard", default="runs/leaderboard.jsonl")
    pr.add_argument("--seed", type=int, default=None)
    pr.add_argument("--run-id", dest="run_id", default=None)
    pr.add_argument("--proxy", default=None, help="factory_mqar | short_horizon_pred")
    pr.add_argument("--max-trials", dest="max_trials", type=int, default=None)
    pr.add_argument("--no-ssl", action="store_true", help="T0: SSL 표현축 고정")
    pr.set_defaults(func=cmd_run)

    pt = sub.add_parser("top", help="리더보드 상위 조회")
    pt.add_argument("--leaderboard", default="runs/leaderboard.jsonl")
    pt.add_argument("-n", type=int, default=5)
    pt.set_defaults(func=cmd_top)

    pe = sub.add_parser("export", help="best config export")
    pe.add_argument("--leaderboard", default="runs/leaderboard.jsonl")
    pe.add_argument("--out", default="runs/export")
    pe.set_defaults(func=cmd_export)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
