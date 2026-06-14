#!/usr/bin/env python3
"""autoresearch-ext — Pi Extension 구현(스윕 시작/중단, best export).

manifest.json의 commands를 실제 동작으로 연결한다. 각 명령은 파이썬 함수이자 CLI로
호출 가능하다. 스윕은 별도 프로세스로 띄우고 pidfile로 추적해 sweep_stop이 실제로
프로세스를 종료한다(데모지만 실동작).

CLI:
    python3 pi-extension/autoresearch_ext.py sweep-start  [--config ...] [--run-id ...]
    python3 pi-extension/autoresearch_ext.py sweep-stop   --run-id <id>
    python3 pi-extension/autoresearch_ext.py export-best  [--out ...]
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
PKG_ROOT = os.path.join(REPO, "tutorial", "autoresearch")
RUNS_DIR = os.path.join(HERE, ".runs")
DEFAULT_CFG = os.path.join(PKG_ROOT, "config", "run_example.yaml")
LB_PATH = os.path.join(PKG_ROOT, "runs", "leaderboard.jsonl")
os.makedirs(RUNS_DIR, exist_ok=True)
sys.path.insert(0, PKG_ROOT)


def _pidfile(run_id: str) -> str:
    safe = "".join(c for c in run_id if c.isalnum() or c in "-_.")
    return os.path.join(RUNS_DIR, f"{safe}.pid")


def _resolve(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(REPO, path)


def sweep_start(config: str = DEFAULT_CFG, run_id: str | None = None) -> dict:
    """ASHA 스윕을 백그라운드 프로세스로 시작."""
    from autoresearch.config_io import load_run_config
    cfg_path = _resolve(config)
    if not os.path.exists(cfg_path):
        return {"status": "error", "error": f"config not found: {cfg_path}"}
    rid = run_id or load_run_config(cfg_path).get("run_id", "run")

    args = [sys.executable, os.path.join(PKG_ROOT, "run.py"), "run",
            "--config", cfg_path, "--leaderboard", LB_PATH]
    log = open(os.path.join(RUNS_DIR, f"{rid}.log"), "w")
    proc = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT, cwd=PKG_ROOT)
    with open(_pidfile(rid), "w") as f:
        f.write(str(proc.pid))
    return {"status": "started", "run_id": rid, "pid": proc.pid,
            "log": os.path.relpath(os.path.join(RUNS_DIR, f"{rid}.log"), REPO)}


def sweep_stop(run_id: str) -> dict:
    """진행 중인 스윕 프로세스 중단."""
    pf = _pidfile(run_id)
    if not os.path.exists(pf):
        return {"status": "not_running", "run_id": run_id}
    pid = int(open(pf).read().strip())
    try:
        os.kill(pid, signal.SIGTERM)
        status = "stopped"
    except ProcessLookupError:
        status = "not_running"
    os.remove(pf)
    return {"status": status, "run_id": run_id, "pid": pid}


def export_best(out: str = os.path.join(PKG_ROOT, "runs", "export")) -> dict:
    """리더보드 best config export(동치성 재확인 포함)."""
    from autoresearch.export import export_best as _export
    try:
        return _export(LB_PATH, _resolve(out))
    except Exception as e:  # noqa: BLE001
        return {"status": "error", "error": str(e)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="autoresearch-ext (Pi Extension)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s1 = sub.add_parser("sweep-start")
    s1.add_argument("--config", default=DEFAULT_CFG)
    s1.add_argument("--run-id", default=None)

    s2 = sub.add_parser("sweep-stop")
    s2.add_argument("--run-id", required=True)

    s3 = sub.add_parser("export-best")
    s3.add_argument("--out", default=os.path.join(PKG_ROOT, "runs", "export"))

    args = ap.parse_args(argv)
    if args.cmd == "sweep-start":
        result = sweep_start(args.config, args.run_id)
    elif args.cmd == "sweep-stop":
        result = sweep_stop(args.run_id)
    else:
        result = export_best(args.out)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
