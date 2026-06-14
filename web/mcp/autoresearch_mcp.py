#!/usr/bin/env python3
"""autoresearch-mcp — AutoResearch 노드를 외부 도구로 노출하는 MCP 서버.

의존성 없이(stdlib만) MCP stdio 트랜스포트(줄 단위 JSON-RPC 2.0)를 직접 구현한다.
공식 `mcp` 패키지가 있으면 그대로 써도 되지만, 무설치 실행을 위해 최소 구현을 둔다.

제공 도구(tools):
  - leaderboard_top      : 상위 trial 목록
  - leaderboard_summary  : 리더보드 요약(best/cost)
  - nas_list_shards      : NAS 매니페스트의 커밋된 샤드 목록
  - run_status           : 최근 스윕 요약(리더보드 기반)
  - export_best          : best config export

Pi/Claude 등 MCP 클라이언트 등록 예(claude_desktop_config.json / pi mcp):
  {"command": "python3", "args": ["web/mcp/autoresearch_mcp.py"]}

자체 점검:  python3 web/mcp/autoresearch_mcp.py --selftest
"""

from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
PKG_ROOT = os.path.join(REPO, "tutorial", "autoresearch")
LB_PATH = os.path.join(PKG_ROOT, "runs", "leaderboard.jsonl")
CFG_PATH = os.path.join(PKG_ROOT, "config", "run_example.yaml")
sys.path.insert(0, PKG_ROOT)

PROTOCOL_VERSION = "2024-11-05"
SERVER_INFO = {"name": "autoresearch-mcp", "version": "0.1.0"}

TOOLS = [
    {
        "name": "leaderboard_top",
        "description": "AutoResearch 리더보드 상위 trial(cfg/proxy/full/cost)을 반환한다.",
        "inputSchema": {
            "type": "object",
            "properties": {"n": {"type": "integer", "description": "개수(기본 5)"}},
        },
    },
    {
        "name": "leaderboard_summary",
        "description": "리더보드 요약(trial 수, 총 학습비용, best)을 반환한다.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "leaderboard_get",
        "description": "trial_id로 리더보드 레코드(최신, last-write-wins)를 조회한다.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "trial_id": {"type": "string"},
                "run_id": {"type": "string", "description": "선택(동일 trial_id 충돌 방지)"},
            },
            "required": ["trial_id"],
        },
    },
    {
        "name": "leaderboard_write",
        "description": "리더보드에 레코드를 append 기록한다(last-write-wins). 외부 워커가 결과를 등록할 때 사용.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "run_id": {"type": "string"},
                "trial_id": {"type": "string"},
                "cfg": {"type": "object"},
                "proxy_score": {"type": "number"},
                "full_score": {"type": "number"},
                "cost": {"type": "number"},
                "rung": {"type": "integer"},
                "status": {"type": "string"},
            },
            "required": ["run_id", "trial_id", "cfg"],
        },
    },
    {
        "name": "nas_list_shards",
        "description": "NAS 매니페스트에서 커밋된(읽기 안전한) 샤드 목록을 반환한다.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "run_status",
        "description": "최근 스윕 상태 요약(리더보드 기반: best/trial 수/백엔드).",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "export_best",
        "description": "best config를 추론/엣지 배포용 번들로 export한다(동치성 재확인 포함).",
        "inputSchema": {
            "type": "object",
            "properties": {"out": {"type": "string", "description": "출력 디렉터리"}},
        },
    },
]


# ── 도구 구현 ────────────────────────────────────────────────────────────────
def _tool_leaderboard_top(args):
    from autoresearch.leaderboard import Leaderboard
    n = int(args.get("n", 5))
    lb = Leaderboard(LB_PATH)
    return [
        {"trial_id": r.trial_id, "proxy_score": r.proxy_score, "full_score": r.full_score,
         "cost": r.cost, "rung": r.rung, "status": r.status, "cfg": r.cfg}
        for r in lb.top(n)
    ]


def _tool_leaderboard_summary(args):
    from autoresearch.leaderboard import Leaderboard
    return Leaderboard(LB_PATH).summary()


def _tool_leaderboard_get(args):
    from autoresearch.leaderboard import Leaderboard
    lb = Leaderboard(LB_PATH)
    tid = args.get("trial_id")
    rid = args.get("run_id")
    for r in lb.latest():
        if r.trial_id == tid and (rid is None or r.run_id == rid):
            return {"trial_id": r.trial_id, "run_id": r.run_id, "cfg": r.cfg,
                    "proxy_score": r.proxy_score, "full_score": r.full_score,
                    "cost": r.cost, "rung": r.rung, "status": r.status,
                    "code_commit": r.code_commit, "seed": r.seed}
    return {"found": False, "trial_id": tid}


def _tool_leaderboard_write(args):
    from autoresearch.leaderboard import Leaderboard, Record
    lb = Leaderboard(LB_PATH)
    rec = Record(
        run_id=args["run_id"], trial_id=args["trial_id"], cfg=args["cfg"],
        proxy_score=args.get("proxy_score"), full_score=args.get("full_score"),
        cost=float(args.get("cost", 0.0)), rung=int(args.get("rung", 0)),
        status=args.get("status", "done"),
    )
    lb.write(rec)
    return {"written": True, "key": rec.key()}


def _tool_nas_list_shards(args):
    from autoresearch.config_io import load_run_config
    from autoresearch.data_interface import NASDataInterface
    cfg = load_run_config(CFG_PATH) if os.path.exists(CFG_PATH) else {}
    nas = NASDataInterface(cfg.get("data", {}).get("manifest", ""),
                           cfg.get("data", {}).get("cache", "/tmp/autoresearch_cache"))
    shards = nas.list_shards(only_committed=True)
    return {"status": nas.cache_status(),
            "shards": [{"shard_id": s.shard_id, "num_samples": s.num_samples,
                        "embedded": s.embedded} for s in shards]}


def _tool_run_status(args):
    from autoresearch.leaderboard import Leaderboard
    from autoresearch.model_adapter import backend_label
    lb = Leaderboard(LB_PATH)
    s = lb.summary()
    s["backend"] = backend_label()
    return s


def _tool_export_best(args):
    from autoresearch.export import export_best
    out = args.get("out") or os.path.join(PKG_ROOT, "runs", "export")
    return export_best(LB_PATH, out)


DISPATCH = {
    "leaderboard_top": _tool_leaderboard_top,
    "leaderboard_summary": _tool_leaderboard_summary,
    "leaderboard_get": _tool_leaderboard_get,
    "leaderboard_write": _tool_leaderboard_write,
    "nas_list_shards": _tool_nas_list_shards,
    "run_status": _tool_run_status,
    "export_best": _tool_export_best,
}


# ── JSON-RPC / MCP 핸들링 ────────────────────────────────────────────────────
def handle(req: dict) -> dict | None:
    mid = req.get("id")
    method = req.get("method")
    params = req.get("params") or {}

    if method == "initialize":
        return _ok(mid, {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {}},
            "serverInfo": SERVER_INFO,
        })
    if method == "notifications/initialized":
        return None  # 알림은 응답 없음
    if method == "tools/list":
        return _ok(mid, {"tools": TOOLS})
    if method == "tools/call":
        name = params.get("name")
        args = params.get("arguments") or {}
        fn = DISPATCH.get(name)
        if fn is None:
            return _err(mid, -32602, f"unknown tool: {name}")
        try:
            result = fn(args)
        except Exception as e:  # noqa: BLE001
            return _ok(mid, {"content": [{"type": "text", "text": f"error: {e}"}],
                             "isError": True})
        text = json.dumps(result, ensure_ascii=False, indent=2)
        return _ok(mid, {"content": [{"type": "text", "text": text}]})
    if method == "ping":
        return _ok(mid, {})
    return _err(mid, -32601, f"method not found: {method}")


def _ok(mid, result):
    return {"jsonrpc": "2.0", "id": mid, "result": result}


def _err(mid, code, msg):
    return {"jsonrpc": "2.0", "id": mid, "error": {"code": code, "message": msg}}


def serve_stdio():
    """줄 단위 JSON-RPC over stdio(MCP stdio 트랜스포트)."""
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue
        resp = handle(req)
        if resp is not None:
            sys.stdout.write(json.dumps(resp, ensure_ascii=False) + "\n")
            sys.stdout.flush()


def selftest():
    """stdio 없이 tools/list + 각 도구 호출을 점검."""
    print("initialize:", handle({"jsonrpc": "2.0", "id": 1, "method": "initialize"})["result"]["serverInfo"])
    tools = handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})["result"]["tools"]
    print("tools:", [t["name"] for t in tools])
    for name in ("leaderboard_summary", "leaderboard_top", "nas_list_shards", "run_status"):
        r = handle({"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                    "params": {"name": name, "arguments": {"n": 2}}})
        txt = r["result"]["content"][0]["text"]
        print(f"\n# {name}\n{txt[:300]}")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        serve_stdio()
