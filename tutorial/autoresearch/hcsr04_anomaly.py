#!/usr/bin/env python3
"""HC-SR04 드리프트/이상 탐지 — 실 센서 데이터(저주파 초음파 거리)로 예측오차 기반 탐지.

설계 §5/§11 "예측오차 이상탐지" 목적을 실 데이터로 실현. 데이터 규모(900행)에 맞춰
딥모델 대신 견고한 통계 탐지기(stdlib)를 쓴다 — 정직한 도구 선택.

방법:
  1) 정상 baseline: 각 설정거리(Dist)의 *첫 세그먼트*에서 robust 중심(median)·산포(MAD→σ).
  2) 점 이상(point anomaly): z=(x-median)/σ, |z|>thr 플래그.
  3) 드리프트(drift): 같은 거리의 후속 세그먼트 평균이 reference에서 유의하게 이동하면 플래그
     (예: set=35 초기→후기 Δ≈-17, 원거리 std↑).
  4) 검증: 합성 이상치(스파이크/드롭아웃) 주입 → 탐지 recall + 실제 드리프트 포착 확인.

사용:
    python3 hcsr04_anomaly.py --csv ../../data/hcsr04/dist_hcsr04.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics as st
from collections import defaultdict


def load(csv_path):
    rows = list(csv.DictReader(open(csv_path)))
    return [(float(r["Time"]), float(r["Dist"]), float(r["Measured"])) for r in rows]


SIGMA_FLOOR = 2.0  # 센서 양자화/노이즈 바닥(raw units) — σ≈0로 인한 오탐 방지


def segments(data):
    """거리(Dist) 변화 기준 연속 구간으로 분할. 반환: [(seg_id, dist, time0, [measured...])].

    설정거리가 바뀔 때마다 새 세그먼트 → 같은 거리가 시간상 떨어져 두 번 나오면 두 세그먼트
    (예: set=35 초기/후기) → 드리프트 비교 가능.
    """
    segs = []
    cur_d = None
    for t, d, m in data:
        if d != cur_d:
            segs.append([len(segs), d, t, []])
            cur_d = d
        segs[-1][3].append(m)
    return [(s[0], s[1], s[2], s[3]) for s in segs]


def mad_sigma(xs):
    med = st.median(xs)
    mad = st.median([abs(x - med) for x in xs])
    return med, 1.4826 * mad


def build_baseline(segs):
    """거리별 첫 세그먼트를 reference 정상으로. σ는 MAD/표준편차/바닥값 중 최대(견고)."""
    base = {}
    for sid, d, t, xs in segs:
        if d not in base:
            med, mad_s = mad_sigma(xs)
            sig = max(mad_s, st.pstdev(xs), SIGMA_FLOOR)
            base[d] = {"median": med, "sigma": sig, "ref_seg": sid, "ref_time": t}
    return base


def detect(segs, base, z_thr=4.0, drift_thr=5.0):
    point_anoms = []
    drifts = []
    for sid, d, t, xs in segs:
        ref = base.get(d)
        if ref is None:
            continue
        med, sig = ref["median"], ref["sigma"]
        # 점 이상
        for i, x in enumerate(xs):
            z = (x - med) / sig
            if abs(z) > z_thr:
                point_anoms.append({"seg": sid, "dist": d, "idx": i, "value": x,
                                    "z": round(z, 2)})
        # 드리프트: 후속 세그먼트 *평균*의 이동을 표준오차(SE=σ/√n) 단위로 검정.
        # (개별점 σ가 아니라 평균의 SE — n개 평균의 작은 이동도 통계적으로 유의할 수 있음)
        if sid != ref["ref_seg"]:
            seg_mean = st.mean(xs)
            se = sig / (len(xs) ** 0.5)
            shift = (seg_mean - med) / se
            if abs(shift) > drift_thr:
                drifts.append({"seg": sid, "dist": d, "time": t, "n": len(xs),
                               "ref_median": round(med, 1), "seg_mean": round(seg_mean, 1),
                               "delta": round(seg_mean - med, 1),
                               "shift_SE": round(shift, 1)})
    return point_anoms, drifts


def inject_anomalies(data, n_spike=10, n_dropout=10, seed=0):
    """검증용 합성 이상치 주입(스파이크 ×1.5, 드롭아웃→40). 주입 위치 라벨 반환."""
    rng = _lcg(seed)
    data = list(data)
    labels = set()
    n = len(data)
    for _ in range(n_spike):
        i = rng(n)
        t, d, m = data[i]
        data[i] = (t, d, m * 1.5)
        labels.add(i)
    for _ in range(n_dropout):
        i = rng(n)
        t, d, m = data[i]
        data[i] = (t, d, 40.0)  # 최소거리 카운트로 드롭
        labels.add(i)
    return data, labels


def _lcg(seed):
    s = {"v": (seed * 2862933555777941757 + 1) & ((1 << 64) - 1)}

    def nxt(n):
        s["v"] = (s["v"] * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return (s["v"] >> 17) % n
    return nxt


def main(argv=None):
    here = os.path.dirname(os.path.abspath(__file__))
    default_csv = os.path.normpath(os.path.join(here, "..", "..", "data", "hcsr04", "dist_hcsr04.csv"))
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=default_csv)
    ap.add_argument("--z-thr", dest="z_thr", type=float, default=4.0)
    ap.add_argument("--drift-thr", dest="drift_thr", type=float, default=5.0)
    args = ap.parse_args(argv)

    data = load(args.csv)
    segs = segments(data)
    base = build_baseline(segs)
    pts, drifts = detect(segs, base, args.z_thr, args.drift_thr)

    # 실데이터 결과
    report = {
        "rows": len(data), "segments": len(segs),
        "distances": sorted(base.keys()),
        "real_point_anomalies": len(pts),
        "real_drifts": drifts,
    }

    # 검증: 합성 이상치 주입 → 탐지 recall
    inj_data, labels = inject_anomalies(data, n_spike=10, n_dropout=10, seed=7)
    inj_segs = segments(inj_data)
    inj_pts, _ = detect(inj_segs, base, args.z_thr, args.drift_thr)
    # 주입 인덱스(전역)와 탐지 매핑: 세그먼트내 idx → 전역 idx 복원
    seg_offsets = {}
    off = 0
    for sid, d, t, xs in inj_segs:
        seg_offsets[sid] = off
        off += len(xs)
    detected_global = {seg_offsets[a["seg"]] + a["idx"] for a in inj_pts}
    tp = len(labels & detected_global)
    recall = tp / len(labels) if labels else 0.0
    fp = len(detected_global - labels)
    report["validation"] = {
        "injected": len(labels), "detected_injected": tp,
        "recall": round(recall, 3), "false_positives": fp,
        "z_thr": args.z_thr,
    }

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
