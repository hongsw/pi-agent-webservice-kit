#!/usr/bin/env python3
"""3자 비교 — 표준 트랜스포머(softmax) vs 선형 병렬 vs 선형 재귀(메모리캐싱).

같은 차원(d_model/n_heads/n_layers)에서 시퀀스 길이 L별 peak GPU 메모리를 측정한다.
  1) Transformer(full)   : 표준 causal softmax MHA — 학습/병렬 형태, L×L 어텐션 → O(L²)
  2) Transformer(kvcache): 증분 디코딩, KV 캐시 유지 → O(L) (길이만큼 증가)
  3) Linear(parallel)    : 우리 선형어텐션 forward (감쇠행렬) → O(L²)
  4) Linear(recurrent)   : 우리 forward_recurrent, 상태 S,z만 → O(1) 상수

정확도: (3)과 (4)는 같은 가중치 → 출력 동치(diff~1e-7, verify_recurrent.py에서 증명).
트랜스포머는 다른 아키텍처(회상 정확도는 더 강하나 추론 메모리는 O(L) — 트레이드오프).

    python3 compare3.py
"""

from __future__ import annotations

import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.nn as nn
import torch.nn.functional as F
from autoresearch.real.model import build_real, RMSNorm

VOCAB = 64
D, H, LAYERS = 256, 8, 4
DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ── 표준 트랜스포머(causal softmax MHA) ──────────────────────────────────────
class TFLayer(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h, self.dh = h, d // h
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.n1, self.n2 = RMSNorm(d), RMSNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def _qkv(self, x):
        B, L, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, -1)
        f = lambda t: t.view(B, L, self.h, self.dh).transpose(1, 2)
        return f(q), f(k), f(v)

    def forward(self, x):  # full O(L^2)
        B, L, _ = x.shape
        q, k, v = self._qkv(self.n1(x))
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        o = o.transpose(1, 2).reshape(B, L, -1)
        x = x + self.o(o)
        return x + self.mlp(self.n2(x))


class Transformer(nn.Module):
    def __init__(self, vocab, d, h, layers, max_len):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d))
        self.layers = nn.ModuleList([TFLayer(d, h) for _ in range(layers)])
        self.norm = RMSNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)

    def forward(self, x):
        h = self.emb(x) + self.pos[:, :x.shape[1]]
        for lyr in self.layers:
            h = lyr(h)
        return self.head(self.norm(h))

    @torch.no_grad()
    def kvcache_decode(self, x):
        """증분 디코딩 — 레이어별 K,V 캐시를 길이만큼 누적(O(L) 메모리)."""
        B, L = x.shape
        caches = [{"k": None, "v": None} for _ in self.layers]
        for t in range(L):
            h = self.emb(x[:, t:t + 1]) + self.pos[:, t:t + 1]
            for li, lyr in enumerate(self.layers):
                q, k, v = lyr._qkv(lyr.n1(h))
                c = caches[li]
                c["k"] = k if c["k"] is None else torch.cat([c["k"], k], dim=2)
                c["v"] = v if c["v"] is None else torch.cat([c["v"], v], dim=2)
                o = F.scaled_dot_product_attention(q, c["k"], c["v"])
                o = o.transpose(1, 2).reshape(B, 1, -1)
                h = h + lyr.o(o)
                h = h + lyr.mlp(lyr.n2(h))
        return caches


def kv_cache_bytes(L):
    return 2 * LAYERS * H * (D // H) * L * 4  # K+V, float32


def peak_reset():
    if DEV == "cuda":
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()


def peak_mb():
    if DEV == "cuda":
        torch.cuda.synchronize(); return round(torch.cuda.max_memory_allocated() / 1e6, 1)
    return -1.0


def run(fn):
    peak_reset()
    try:
        fn(); return peak_mb()
    except RuntimeError as e:
        torch.cuda.empty_cache()
        return "OOM" if "out of memory" in str(e).lower() else "ERR"


def main():
    torch.manual_seed(0)
    cfg = {"base_rule": "dla", "aggregation": "residual", "d_model": D,
           "n_layers": LAYERS, "n_heads": H, "segment_len": 64,
           "segmentation": "logarithmic", "init_mode": "independent",
           "ssl": {"encoder": "none", "invariance_coeff": "low"}}
    maxL = 40000
    lin = build_real(cfg, VOCAB, max_len=maxL).to(DEV).eval()
    tf = Transformer(VOCAB, D, H, LAYERS, max_len=maxL).to(DEV).eval()
    print(f"device={DEV} d_model={D} heads={H} layers={LAYERS}")
    print(f"{'L':>7} {'TF(full)':>10} {'Lin(par)':>10} {'Lin(rec)':>10} "
          f"{'TF KVcache':>12} {'Rec state':>10}")
    rows = []
    for L in [512, 1024, 2048, 4096, 8192, 16384, 32768]:
        x = torch.randint(1, VOCAB, (1, L), device=DEV)
        with torch.no_grad():
            tf_full = run(lambda: tf(x))
            lin_par = run(lambda: lin(x))
            lin_rec = run(lambda: lin.forward_recurrent(x)) if L <= 8192 else "skip(t)"
        row = {"L": L, "tf_full_mb": tf_full, "lin_parallel_mb": lin_par,
               "lin_recurrent_mb": lin_rec,
               "tf_kvcache_bytes": kv_cache_bytes(L),
               "rec_state_bytes": lin.state_bytes(1)}
        rows.append(row)
        print(f"{L:>7} {str(tf_full):>10} {str(lin_par):>10} {str(lin_rec):>10} "
              f"{kv_cache_bytes(L)//1024:>10}KB {lin.state_bytes(1)//1024:>8}KB")
    out = {"device": DEV, "dims": {"d": D, "h": H, "layers": LAYERS}, "scan": rows}
    open(os.path.join(os.path.dirname(__file__), "compare3_result.json"), "w").write(
        json.dumps(out, ensure_ascii=False, indent=2))
    print("\nKV캐시(트랜스포머 추론)는 L에 비례 증가, 재귀 상태는 상수 →",
          f"L=16384에서 {kv_cache_bytes(16384)//1024}KB vs {lin.state_bytes(1)//1024}KB "
          f"(~{kv_cache_bytes(16384)//lin.state_bytes(1)}×)")


if __name__ == "__main__":
    main()
