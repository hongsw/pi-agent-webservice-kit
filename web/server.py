#!/usr/bin/env python3
"""Web UI — AutoResearch 리더보드 대시보드 + 스윕 런처.

순수 stdlib(http.server)로 구현 → 무설치 실행:
    python3 web/server.py            # http://localhost:8765
    PORT=9000 python3 web/server.py

엔드포인트:
    GET  /                      대시보드(HTML)
    GET  /api/summary           리더보드 요약 + NAS 캐시 상태 + 백엔드
    GET  /api/leaderboard?n=20  상위 trial 목록
    GET  /api/runs              진행 중/완료 스윕 상태
    POST /api/sweep             스윕 시작(백그라운드 스레드)  body: {"config": "...", "run_id": "..."}
    POST /api/export            best config export

CLI 전용 금지 요구사항을 충족하는 사용자 화면. Pi Extension(autoresearch-ext)과
MCP(autoresearch-mcp)가 이 REST/리더보드를 공유한다.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
PKG_ROOT = os.path.join(REPO, "tutorial", "autoresearch")
DEFAULT_LB = os.path.join(PKG_ROOT, "runs", "leaderboard.jsonl")
DEFAULT_CFG = os.path.join(PKG_ROOT, "config", "run_example.yaml")
STATIC = os.path.join(HERE, "static")

sys.path.insert(0, PKG_ROOT)
from autoresearch.config_io import load_run_config       # noqa: E402
from autoresearch.export import export_best               # noqa: E402
from autoresearch.leaderboard import Leaderboard          # noqa: E402
from autoresearch.loop import autoresearch_loop           # noqa: E402

# 진행 중 스윕 상태(메모리). 데모용 — 영속 상태는 리더보드(JSONL).
_RUNS: dict[str, dict] = {}
_RUNS_LOCK = threading.Lock()


def _launch_sweep(config_path: str, run_id: str | None) -> str:
    run_cfg = load_run_config(config_path)
    if run_id:
        run_cfg["run_id"] = run_id
    rid = run_cfg.get("run_id", f"run_{int(time.time())}")
    logs: list[str] = []

    with _RUNS_LOCK:
        _RUNS[rid] = {"run_id": rid, "status": "running", "started": time.time(),
                      "logs": logs, "summary": None}

    def worker():
        try:
            summary = autoresearch_loop(run_cfg, DEFAULT_LB, log=lambda m: logs.append(m))
            with _RUNS_LOCK:
                _RUNS[rid].update(status="done", summary=summary, finished=time.time())
        except Exception as e:  # noqa: BLE001
            with _RUNS_LOCK:
                _RUNS[rid].update(status="failed", error=str(e), finished=time.time())

    threading.Thread(target=worker, daemon=True).start()
    return rid


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # 조용히
        pass

    def _send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200):
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path == "/" or u.path == "/index.html":
            try:
                body = open(os.path.join(STATIC, "index.html"), "rb").read()
            except FileNotFoundError:
                body = b"<h1>index.html missing</h1>"
            return self._send(200, body, "text/html; charset=utf-8")
        if u.path == "/api/summary":
            return self._json(self._summary())
        if u.path == "/api/leaderboard":
            n = int(q.get("n", ["20"])[0])
            lb = Leaderboard(DEFAULT_LB)
            rows = [self._row(r) for r in lb.top(n)]
            return self._json({"rows": rows})
        if u.path == "/api/runs":
            with _RUNS_LOCK:
                runs = [{k: v for k, v in r.items() if k != "logs"} for r in _RUNS.values()]
                # 마지막 로그 몇 줄만
                for r, src in zip(runs, _RUNS.values()):
                    r["last_logs"] = src["logs"][-6:]
            return self._json({"runs": sorted(runs, key=lambda x: x.get("started", 0), reverse=True)})
        return self._json({"error": "not found"}, 404)

    def do_POST(self):
        u = urlparse(self.path)
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            return self._json({"error": "invalid json"}, 400)

        if u.path == "/api/sweep":
            cfg = body.get("config") or DEFAULT_CFG
            if not os.path.isabs(cfg):
                cfg = os.path.join(PKG_ROOT, cfg)
            if not os.path.exists(cfg):
                return self._json({"error": f"config not found: {cfg}"}, 400)
            rid = _launch_sweep(cfg, body.get("run_id"))
            return self._json({"started": rid})
        if u.path == "/api/export":
            out = body.get("out") or os.path.join(PKG_ROOT, "runs", "export")
            try:
                result = export_best(DEFAULT_LB, out)
            except Exception as e:  # noqa: BLE001
                return self._json({"error": str(e)}, 400)
            return self._json(result)
        return self._json({"error": "not found"}, 404)

    # ── helpers ──
    def _summary(self):
        from autoresearch.data_interface import NASDataInterface
        from autoresearch.model_adapter import backend_label
        lb = Leaderboard(DEFAULT_LB)
        cfg = load_run_config(DEFAULT_CFG) if os.path.exists(DEFAULT_CFG) else {}
        nas = NASDataInterface(cfg.get("data", {}).get("manifest", ""),
                               cfg.get("data", {}).get("cache", "/tmp/autoresearch_cache"))
        return {
            "backend": backend_label(),
            "leaderboard": lb.summary(),
            "nas": nas.cache_status(),
            "default_config": os.path.relpath(DEFAULT_CFG, REPO),
        }

    @staticmethod
    def _row(r):
        return {
            "trial_id": r.trial_id, "run_id": r.run_id,
            "proxy_score": r.proxy_score, "full_score": r.full_score,
            "cost": r.cost, "rung": r.rung, "status": r.status,
            "backend": r.backend, "cfg": r.cfg,
        }


def main():
    port = int(os.environ.get("PORT", "8765"))
    httpd = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print(f"[web] AutoResearch 대시보드: http://localhost:{port}")
    print(f"[web] leaderboard: {DEFAULT_LB}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[web] bye")


if __name__ == "__main__":
    main()
