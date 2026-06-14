"""prepare.py — 불변(데이터 준비 + 평가). karpathy/autoresearch의 prepare.py에 대응.

★ 이 파일은 수정하지 않는다. 모든 실험이 동일한 기준(val_bpb)으로 평가되도록 보장한다.

mock: torch/GPU 없이 동작하도록, '검증 데이터의 숨은 구조(VAL_SECRET)'에 모델 knob이
얼마나 가까운지로 val_bpb를 계산한다. 실물에서는 여기서 검증셋을 로드하고 모델의
bits-per-byte를 직접 측정한다(낮을수록 우수, vocab 크기 독립).
"""

from __future__ import annotations

import hashlib
import math
from typing import Any

# 검증 데이터가 '선호하는' 모델 구조 — 학습으로는 알 수 없는 데이터의 물리.
VAL_SECRET = {"width": 512, "depth": 8, "dropout": 0.10, "lr": 3.0e-3}

# val_bpb 범위(대략): 최적이면 ~0.95, 멀수록 ~2.5+.
_BPB_FLOOR = 0.95
_BPB_SCALE = 1.6


def _norm_dist(model: dict[str, Any]) -> float:
    """모델 knob과 VAL_SECRET 간 정규화 거리(0=완벽)."""
    def rel(key: str, lo: float, hi: float) -> float:
        v = float(model.get(key, lo))
        t = float(VAL_SECRET[key])
        span = (hi - lo) or 1.0
        return abs(v - t) / span

    d = 0.0
    d += rel("width", 64, 1024) ** 2
    d += rel("depth", 1, 24) ** 2
    d += rel("dropout", 0.0, 0.6) ** 2
    # lr은 로그 스케일 비교
    lr = max(1e-5, float(model.get("lr", 1e-3)))
    d += (abs(math.log10(lr) - math.log10(VAL_SECRET["lr"])) / 3.0) ** 2
    return math.sqrt(d / 4.0)


def _eval_noise(model: dict[str, Any]) -> float:
    """평가 노이즈(결정적) — 같은 모델은 같은 val_bpb. seed처럼 동작."""
    blob = repr(sorted(model.items())).encode("utf-8")
    h = int(hashlib.sha1(blob).hexdigest(), 16)
    return ((h % 1000) / 1000.0 - 0.5) * 0.04  # ±0.02


def evaluate(model: dict[str, Any]) -> float:
    """모델을 받아 val_bpb 반환(낮을수록 우수)."""
    bpb = _BPB_FLOOR + _BPB_SCALE * _norm_dist(model) + _eval_noise(model)
    return round(bpb, 4)


def load_val_info() -> dict[str, Any]:
    """검증셋 메타(flavor). 실물에선 실제 검증 바이트/토큰 수."""
    return {"val_bytes": 1_048_576, "metric": "val_bpb (lower is better)"}
