#!/usr/bin/env python3
"""NAS 도커 서버 — 커밋된 샤드 매니페스트 + 샤드 파일을 HTTP로 제공(설계 §7).

수집기가 커밋한 샤드만 노출한다(쓰는 중 파일 미접근 = committed=false는 매니페스트에서 제외).
학습 노드(autoresearch)는 manifest URL을 읽어 커밋 샤드를 read한다 — 실제 연결.

엔드포인트:
    GET /health
    GET /manifest            커밋된 샤드 jsonl (data_interface가 파싱)
    GET /shards/<file>       개별 샤드(jsonl)
    POST /commit             (데모) 새 샤드 1개 생성·커밋 — 수집 시뮬레이션

환경: PORT(기본 8080), DATA_DIR(기본 /data), SEED_SHARDS(기본 8), SAMPLES_PER_SHARD(기본 256).
순수 stdlib — torch/외부 의존 없음.
"""

from __future__ import annotations

import hashlib
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

PORT = int(os.environ.get("PORT", "8080"))
DATA_DIR = os.environ.get("DATA_DIR", "/data")
PUBLIC_BASE = os.environ.get("PUBLIC_BASE", f"http://localhost:{PORT}")
SEED_SHARDS = int(os.environ.get("SEED_SHARDS", "8"))
SAMPLES = int(os.environ.get("SAMPLES_PER_SHARD", "256"))
VOCAB = int(os.environ.get("VOCAB", "64"))

SHARD_DIR = os.path.join(DATA_DIR, "shards")
MANIFEST = os.path.join(DATA_DIR, "manifest.jsonl")


def _rng(seed: int):
    """의존성 없는 결정적 의사난수(LCG)."""
    state = {"s": (seed * 6364136223846793005 + 1) & ((1 << 64) - 1)}

    def nxt(n: int) -> int:
        state["s"] = (state["s"] * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return (state["s"] >> 17) % n
    return nxt


def _make_shard(idx: int) -> dict:
    """합성 factory 샤드 1개 생성(유닛/이벤트 시퀀스). 커밋 상태로 기록."""
    rnd = _rng(1000 + idx)
    path = os.path.join(SHARD_DIR, f"shard_{idx:04d}.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for j in range(SAMPLES):
            # 공장 단위(unit)/이벤트(event)/결함클래스(defect) 합성 레코드
            rec = {
                "unit_id": rnd(200),
                "event": rnd(VOCAB),
                "defect_class": rnd(6),
                "t": j,
            }
            f.write(json.dumps(rec) + "\n")
    return {
        "shard_id": f"shard_{idx:04d}",
        "path": f"{PUBLIC_BASE}/shards/shard_{idx:04d}.jsonl",
        "num_samples": SAMPLES,
        "committed": True,
        "embedded": False,
    }


def seed_if_empty():
    os.makedirs(SHARD_DIR, exist_ok=True)
    if os.path.exists(MANIFEST):
        return
    rows = [_make_shard(i) for i in range(SEED_SHARDS)]
    with open(MANIFEST, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"[nas] seeded {len(rows)} committed shards ({SAMPLES} samples each) → {DATA_DIR}")


def append_commit() -> dict:
    """새 샤드를 생성·커밋(수집기 시뮬레이션). last-write 매니페스트에 append."""
    existing = 0
    if os.path.exists(MANIFEST):
        existing = sum(1 for _ in open(MANIFEST))
    row = _make_shard(existing)
    with open(MANIFEST, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    return row


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body: bytes, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        p = urlparse(self.path).path
        if p == "/health":
            return self._send(200, b'{"status":"ok","role":"nas"}')
        if p == "/manifest":
            data = open(MANIFEST, "rb").read() if os.path.exists(MANIFEST) else b""
            return self._send(200, data, "application/x-ndjson")
        if p.startswith("/shards/"):
            fn = os.path.basename(p)
            fp = os.path.join(SHARD_DIR, fn)
            if os.path.exists(fp):
                return self._send(200, open(fp, "rb").read(), "application/x-ndjson")
            return self._send(404, b'{"error":"shard not found"}')
        return self._send(404, b'{"error":"not found"}')

    def do_POST(self):
        p = urlparse(self.path).path
        if p == "/commit":
            row = append_commit()
            return self._send(200, json.dumps({"committed": row}).encode())
        return self._send(404, b'{"error":"not found"}')


def main():
    seed_if_empty()
    httpd = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"[nas] serving committed shards on :{PORT} (manifest=/manifest)")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
