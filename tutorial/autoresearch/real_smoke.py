#!/usr/bin/env python3
"""실물 백엔드 스모크 — recall 학습 곡선으로 학습 속도/분리를 본다.

각 config를 체크포인트(누적 스텝)마다 평가해 recall이 chance 위로 언제 오르는지 확인.
ASHA rung 예산을 정하는 근거가 된다.
    python3 real_smoke.py
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from autoresearch.real.trainer import RealRun  # noqa: E402

RT = {"device": "cuda", "task": "factory_mqar", "vocab": 40, "seq_len": 24,
      "num_pairs": 4, "batch": 64, "lr": 3e-3, "full_seq_len": 36, "seed": 0}
CKPTS = [150, 300, 600, 1000]

BASE = {"segmentation": "logarithmic", "init_mode": "checkpoint",
        "segment_len": 64, "d_model": 256, "n_layers": 4, "n_heads": 8,
        "ssl": {"encoder": "vjepa", "invariance_coeff": "low",
                "positive_pair": "same_defect_class"}}
CONFIGS = [
    {"base_rule": "swla",   "aggregation": "residual"},
    {"base_rule": "linear", "aggregation": "residual"},
    {"base_rule": "dla",    "aggregation": "residual"},
    {"base_rule": "titans", "aggregation": "residual"},
    {"base_rule": "linear", "aggregation": "ssc", "top_k": 4},
    {"base_rule": "linear", "aggregation": "soup"},
]

print(f"chance recall ≈ {1.0 / (RT['vocab'] // 2):.3f}   ckpts={CKPTS}")
for c in CONFIGS:
    cfg = {**BASE, **c}
    run = RealRun(cfg, RT)
    prev = 0
    curve = []
    t0 = time.time()
    for ck in CKPTS:
        run.train(ck - prev, None)
        prev = ck
        recall, ce = run._eval(hard=False)
        curve.append(f"{recall:.3f}")
    dt = time.time() - t0
    print(f"{c['base_rule']:>7}+{c['aggregation']:<8} recall[{'/'.join(curve)}] "
          f"params={run.param_millions():.2f}M ({dt:.1f}s)")
