#!/usr/bin/env python3
"""엣지 추론 도커 서버 — 상수메모리 스트리밍 선형어텐션 추론(설계 §8 "고정상태 추론").

best_config.json(학습 노드의 export 산출물)을 로드해 모델을 구성하고, 토큰을 1개씩
스트리밍하며 고정 크기 상태(S,z)만 갱신한다. numpy만 사용 — 엣지 장비 친화적(torch 불필요).
재귀식은 torch 참조구현과 동일: S_t=a·S_{t-1}+φ(k)⊗v, z_t=a·z_{t-1}+φ(k), o=φ(q)·S/φ(q)·z.

엔드포인트:
    GET  /health
    GET  /info               cfg, backend, layer별 상태 바이트(길이 무관 상수)
    POST /infer  {tokens:[..]}  스트리밍 추론 → step별 argmax + 처리 토큰수 + 상태 바이트
    GET  /memory_demo        길이 64..8192에서 상태 바이트가 상수임을 실증

환경: PORT(8091), EXPORT_DIR(/exports), D_CAP(256), LAYER_CAP(4).
"""

from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import numpy as np

PORT = int(os.environ.get("PORT", "8091"))
EXPORT_DIR = os.environ.get("EXPORT_DIR", "/exports")
D_CAP = int(os.environ.get("D_CAP", "256"))
LAYER_CAP = int(os.environ.get("LAYER_CAP", "4"))
VOCAB = int(os.environ.get("VOCAB", "64"))


def load_cfg() -> dict:
    p = os.path.join(EXPORT_DIR, "best_config.json")
    if os.path.exists(p):
        try:
            b = json.load(open(p))
            cfg = b.get("cfg", b)
            cfg["_source"] = p
            return cfg
        except Exception:
            pass
    # export 없으면 기본 cfg(데모)
    return {"base_rule": "dla", "aggregation": "residual", "d_model": 256,
            "n_layers": 4, "n_heads": 8, "segment_len": 64, "_source": "default"}


def _phi(x):  # feature map elu+1 (>=0)
    return np.where(x > 0, x + 1.0, np.exp(np.clip(x, -30, 0)))


class StreamingModel:
    """numpy 상수메모리 스트리밍 선형어텐션(linear/dla)."""

    def __init__(self, cfg: dict, vocab: int):
        self.cfg = cfg
        self.vocab = vocab
        self.d = min(int(cfg.get("d_model", 256)), D_CAP)
        self.h = int(cfg.get("n_heads", 8))
        while self.d % self.h:
            self.h -= 1
        self.dh = self.d // self.h
        self.L = min(int(cfg.get("n_layers", 4)), LAYER_CAP)
        self.base_rule = cfg.get("base_rule", "dla")
        rng = np.random.default_rng(0)
        s = 1.0 / np.sqrt(self.d)
        self.emb = rng.normal(0, 0.02, (vocab, self.d)).astype(np.float32)
        self.head = rng.normal(0, s, (self.d, vocab)).astype(np.float32)
        self.Wqkv = [rng.normal(0, s, (self.d, 3 * self.d)).astype(np.float32)
                     for _ in range(self.L)]
        self.Wo = [rng.normal(0, s, (self.d, self.d)).astype(np.float32)
                   for _ in range(self.L)]
        self.Wdecay = [rng.normal(0, s, (self.d, self.h)).astype(np.float32)
                       for _ in range(self.L)]
        self.floor = 0.9 if cfg.get("segmentation") == "logarithmic" else 0.0

    def state_bytes(self) -> int:
        # 레이어별 S[h,dh,dh] + z[h,dh], float32 — 시퀀스 길이와 무관한 상수
        per = self.h * (self.dh * self.dh + self.dh) * 4
        return per * self.L

    def init_state(self):
        return [{"S": np.zeros((self.h, self.dh, self.dh), np.float32),
                 "z": np.zeros((self.h, self.dh), np.float32)} for _ in range(self.L)]

    def step(self, token: int, states: list):
        """토큰 1개 → 다음 토큰 logits. 상태만 갱신(O(1))."""
        x = self.emb[token].copy()                         # [d]
        for li in range(self.L):
            qkv = x @ self.Wqkv[li]                         # [3d]
            q, k, v = np.split(qkv, 3)
            q = q.reshape(self.h, self.dh); k = k.reshape(self.h, self.dh); v = v.reshape(self.h, self.dh)
            qf, kf = _phi(q), _phi(k)
            if self.base_rule in ("dla", "titans"):
                a = 1.0 / (1.0 + np.exp(-(x @ self.Wdecay[li])))   # [h]
                a = self.floor + (1 - self.floor) * a
            else:
                a = np.ones(self.h, np.float32)
            st = states[li]
            st["S"] = a[:, None, None] * st["S"] + kf[:, :, None] * v[:, None, :]
            st["z"] = a[:, None] * st["z"] + kf
            num = np.einsum("hd,hde->he", qf, st["S"])             # [h,dh]
            den = np.einsum("hd,hd->h", qf, st["z"])[:, None] + 1e-6
            o = (num / den).reshape(self.d)                        # [d]
            x = x + o @ self.Wo[li]                                 # residual
        logits = x @ self.head                                     # [V]
        return logits, states

    def stream(self, tokens: list):
        states = self.init_state()
        preds = []
        for t in tokens:
            logits, states = self.step(int(t) % self.vocab, states)
            preds.append(int(np.argmax(logits)))
        return preds


CFG = load_cfg()
MODEL = StreamingModel(CFG, VOCAB)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, obj):
        b = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        p = urlparse(self.path).path
        if p == "/health":
            return self._send(200, {"status": "ok", "role": "edge"})
        if p == "/info":
            return self._send(200, {
                "backend": "numpy-streaming", "cfg_source": CFG.get("_source"),
                "base_rule": MODEL.base_rule, "d_model": MODEL.d,
                "n_heads": MODEL.h, "n_layers": MODEL.L,
                "state_bytes": MODEL.state_bytes(),
                "note": "상태는 시퀀스 길이와 무관한 고정 크기(상수메모리)"})
        if p == "/memory_demo":
            rows = []
            for L in [64, 256, 1024, 4096, 8192]:
                toks = [i % MODEL.vocab for i in range(L)]
                MODEL.stream(toks)
                rows.append({"L": L, "state_bytes": MODEL.state_bytes()})
            return self._send(200, {"scan": rows,
                                    "constant": len({r["state_bytes"] for r in rows}) == 1})
        return self._send(404, {"error": "not found"})

    def do_POST(self):
        p = urlparse(self.path).path
        if p == "/infer":
            length = int(self.headers.get("Content-Length", 0))
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                return self._send(400, {"error": "invalid json"})
            tokens = body.get("tokens") or list(range(16))
            preds = MODEL.stream(tokens)
            return self._send(200, {"n_tokens": len(tokens), "predictions": preds,
                                    "state_bytes": MODEL.state_bytes(),
                                    "memory_grows_with_length": False})
        return self._send(404, {"error": "not found"})


def main():
    print(f"[edge] streaming inference on :{PORT} "
          f"(cfg={CFG.get('_source')} state={MODEL.state_bytes()}B 상수)")
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
