#!/usr/bin/env python3
"""재귀(RNN) 추론 검증 — 병렬↔재귀 동치성 + 상수 상태/메모리 캐싱 이점.

설계 §8(엣지 고정상태 추론) · §3.4(동치성 테스트)를 실측한다:
  1. 동치성: forward(병렬, O(L²)) vs forward_recurrent(재귀, O(1) 상태) 출력 max|diff|.
  2. 메모리: 길이 L을 키우며 peak GPU 메모리 — 병렬은 L×L로 폭증/OOM, 재귀는 ~O(L)+상수 상태.

사용:
    python3 verify_recurrent.py --rule linear --json
    python3 verify_recurrent.py --rule dla --bench --json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch  # noqa: E402
from autoresearch.real.model import build_real  # noqa: E402

VOCAB = 64
BASE = {"aggregation": "residual", "segmentation": "logarithmic",
        "init_mode": "independent", "segment_len": 64,
        "d_model": 256, "n_layers": 4, "n_heads": 8,
        "ssl": {"encoder": "none", "invariance_coeff": "low"}}


def _peak_mb_reset():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def _peak_mb():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        return round(torch.cuda.max_memory_allocated() / 1e6, 1)
    return -1.0


def equivalence(rule: str, device: str) -> dict:
    cfg = {**BASE, "base_rule": rule}
    torch.manual_seed(0)
    model = build_real(cfg, VOCAB, max_len=512).to(device).eval()
    x = torch.randint(1, VOCAB, (2, 96), device=device)
    with torch.no_grad():
        y_par = model(x)
        y_rec = model.forward_recurrent(x)
    diff = (y_par - y_rec).abs().max().item()
    rel = diff / (y_par.abs().max().item() + 1e-9)
    return {"max_abs_diff": diff, "max_rel_diff": rel,
            "pass": diff < 1e-3, "is_rnn": model.blocks[0].mix.is_rnn,
            "state_bytes_B1": model.state_bytes(1)}


def stream_check(rule: str, device: str) -> dict:
    """토큰 스트리밍(step 단위) vs 병렬 forward 동치 — 엣지 실시간 추론 경로 검증.

    스트리밍은 init_mode=independent에서만 유효(checkpoint는 전체 컨텍스트 평균 필요).
    """
    cfg = {**BASE, "base_rule": rule, "init_mode": "independent"}
    torch.manual_seed(0)
    model = build_real(cfg, VOCAB, max_len=512).to(device).eval()
    x = torch.randint(1, VOCAB, (2, 80), device=device)
    with torch.no_grad():
        y_par = model(x)
        y_stream = model.stream(x)
    diff = (y_par - y_stream).abs().max().item()
    return {"max_abs_diff": diff, "pass": diff < 1e-3, "L": 80}


def stress(rule: str, device: str) -> dict:
    """동치성 강건성 — aggregation×segmentation×batch×길이를 바꿔도 병렬=재귀인지."""
    worst = 0.0
    cases = []
    for agg in ["residual", "grm", "soup", "ssc"]:
        for seg in ["constant", "logarithmic"]:
            cfg = {**BASE, "base_rule": rule, "aggregation": agg, "segmentation": seg,
                   "top_k": 4, "init_mode": "checkpoint"}
            torch.manual_seed(1)
            model = build_real(cfg, VOCAB, max_len=512).to(device).eval()
            x = torch.randint(1, VOCAB, (4, 320), device=device)   # batch=4, L=320
            with torch.no_grad():
                d = (model(x) - model.forward_recurrent(x)).abs().max().item()
            worst = max(worst, d)
            cases.append({"agg": agg, "seg": seg, "max_diff": d})
    return {"worst_max_diff": worst, "pass": worst < 1e-3, "cases": cases}


def bench(rule: str, device: str) -> dict:
    cfg = {**BASE, "base_rule": rule}
    torch.manual_seed(0)
    model = build_real(cfg, VOCAB, max_len=40000).to(device).eval()
    rows = []
    for L in [512, 1024, 2048, 4096, 8192, 16384]:
        x = torch.randint(1, VOCAB, (1, L), device=device)
        row = {"L": L}
        # 병렬(O(L²)) — OOM 가능
        try:
            _peak_mb_reset()
            with torch.no_grad():
                model(x)
            row["parallel_mb"] = _peak_mb()
        except RuntimeError as e:
            row["parallel_mb"] = "OOM" if "out of memory" in str(e).lower() else f"ERR"
            torch.cuda.empty_cache()
        # 재귀(O(1) 상태) — 시간 절약 위해 L<=4096만 측정(상태 크기는 길이 무관 상수)
        if L <= 4096:
            _peak_mb_reset()
            t0 = time.time()
            with torch.no_grad():
                model.forward_recurrent(x)
            row["recurrent_mb"] = _peak_mb()
            row["recurrent_s"] = round(time.time() - t0, 2)
        else:
            row["recurrent_mb"] = "skip(time)"
        rows.append(row)
    return {"state_bytes_B1": model.state_bytes(1), "scan": rows}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rule", required=True, choices=["linear", "swla", "dla", "titans"])
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--stress", action="store_true")
    ap.add_argument("--stream", action="store_true")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"

    out = {"rule": args.rule, "device": device,
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
           "equivalence": equivalence(args.rule, device)}
    if args.stress:
        out["stress"] = stress(args.rule, device)
    if args.stream:
        out["stream"] = stream_check(args.rule, device)
    if args.bench:
        out["bench"] = bench(args.rule, device)

    if args.json:
        print(json.dumps(out, ensure_ascii=False))
    else:
        eq = out["equivalence"]
        print(f"[{args.rule}] equiv pass={eq['pass']} max_diff={eq['max_abs_diff']:.2e} "
              f"state(B1)={eq['state_bytes_B1']}B is_rnn={eq['is_rnn']}")
        if args.bench:
            print(f"  state_bytes(B1)={out['bench']['state_bytes_B1']}")
            for r in out["bench"]["scan"]:
                print(f"  L={r['L']:>6} parallel={r['parallel_mb']} "
                      f"recurrent={r['recurrent_mb']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
