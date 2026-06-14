"""§3 탐색 공간 — 시퀀스 축(growing-memory 4축) × 표현 축(SSL) × 소형 하이퍼파라미터.

config는 단순 dict로 표현한다. 컨트롤러가 이 공간에서 샘플하고, validity_gate가
학습 전에 유효성을 검증한다. 이론 가지치기(§3.4)는 sample_config에서 적용한다.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from typing import Any


# ── §3.1 시퀀스 축 (growing-memory 4축) ──────────────────────────────────────
BASE_RULE = ["linear", "swla", "dla", "titans"]
AGGREGATION = ["residual", "grm", "soup", "ssc"]
SEGMENTATION = ["constant", "logarithmic"]
INIT_MODE = ["checkpoint", "independent"]
SEGMENT_LEN = [64, 128, 256, 512]
TOP_K = [2, 4, 8]  # ssc 전용

# ── §3.2 표현 축 (SSL) ───────────────────────────────────────────────────────
ENCODER = ["vjepa", "dinov2", "vicreg"]
INVARIANCE_COEFF = ["low", "high"]
POSITIVE_PAIR = ["consecutive_frame", "same_unit_multiview", "same_defect_class"]

# ── §3.3 소형 하이퍼파라미터 ─────────────────────────────────────────────────
D_MODEL = [512, 768, 1024]
N_LAYERS = [6, 12, 24]
N_HEADS = [8, 12, 16]


@dataclass
class SearchSpace:
    """run config(§9)의 `space` 섹션에서 만든 탐색 공간.

    값이 None인 축은 기본 후보 전체를 사용한다(예: SSL 고정 모드 → ssl=None).
    """

    base_rule: list[str] = field(default_factory=lambda: list(BASE_RULE))
    aggregation: list[str] = field(default_factory=lambda: list(AGGREGATION))
    segmentation: list[str] = field(default_factory=lambda: list(SEGMENTATION))
    init_mode: list[str] = field(default_factory=lambda: list(INIT_MODE))
    segment_len: list[int] = field(default_factory=lambda: [128, 256, 512])
    top_k: list[int] = field(default_factory=lambda: list(TOP_K))
    d_model: list[int] = field(default_factory=lambda: list(D_MODEL))
    n_layers: list[int] = field(default_factory=lambda: list(N_LAYERS))
    n_heads: list[int] = field(default_factory=lambda: list(N_HEADS))
    ssl: dict[str, list[str]] | None = None  # None이면 SSL 고정(T0 단계)

    @classmethod
    def from_dict(cls, space: dict[str, Any]) -> "SearchSpace":
        ssl = space.get("ssl")
        return cls(
            base_rule=space.get("base_rule", list(BASE_RULE)),
            aggregation=space.get("aggregation", list(AGGREGATION)),
            segmentation=space.get("segmentation", list(SEGMENTATION)),
            init_mode=space.get("init_mode", list(INIT_MODE)),
            segment_len=space.get("segment_len", [128, 256, 512]),
            top_k=space.get("top_k", list(TOP_K)),
            d_model=space.get("d_model", list(D_MODEL)),
            n_layers=space.get("n_layers", list(N_LAYERS)),
            n_heads=space.get("n_heads", list(N_HEADS)),
            ssl=ssl,
        )

    def grid_size(self) -> int:
        """전수 탐색 시 후보 수(가지치기 전). 탐색 규모 감 잡기용."""
        n = (
            len(self.base_rule)
            * len(self.aggregation)
            * len(self.segmentation)
            * len(self.init_mode)
            * len(self.segment_len)
            * len(self.d_model)
            * len(self.n_layers)
            * len(self.n_heads)
        )
        if self.ssl:
            n *= (
                len(self.ssl.get("encoder", ENCODER))
                * len(self.ssl.get("invariance_coeff", INVARIANCE_COEFF))
                * len(self.ssl.get("positive_pair", POSITIVE_PAIR))
            )
        return n


def _theory_prune_ssl(cfg: dict[str, Any], rng: random.Random) -> None:
    """§3.4 이론 가지치기 — Balestriero·LeCun(2205.11508) closed-form 가이드.

    무작정 전수 탐색하지 않는다. 두 가지 사전 지식을 약한 제약으로 반영:
      - VICReg + pairwise 어긋남 위험이 큰 positive_pair → 낮은 invariance 선호
      - 저데이터/정렬 잘된 pair(consecutive_frame) → 높은 invariance 선호
    완전 강제는 아니고, 일정 확률로 어긋난 조합을 교정해 탐색을 사전 축소한다.
    """
    ssl = cfg.get("ssl")
    if not ssl:
        return
    pair = ssl.get("positive_pair")
    enc = ssl.get("encoder")
    # same_defect_class는 클래스 내 변이가 커 invariance를 높게 주면 붕괴 위험 → low 선호
    if pair == "same_defect_class" and rng.random() < 0.8:
        ssl["invariance_coeff"] = "low"
    # consecutive_frame은 거의 동일 view → high invariance가 표현 안정에 유리
    elif pair == "consecutive_frame" and rng.random() < 0.7:
        ssl["invariance_coeff"] = "high"
    # VICReg은 invariance 항이 직접 노출 → 어긋난 pair에서 low로 당겨줌
    if enc == "vicreg" and pair == "same_unit_multiview" and rng.random() < 0.5:
        ssl["invariance_coeff"] = "low"


def sample_config(space: SearchSpace, rng: random.Random) -> dict[str, Any]:
    """탐색 공간에서 config 한 점을 샘플한다(이론 가지치기 포함).

    반환 dict는 build(cfg) / validity_gate(cfg)가 그대로 받는 평탄한 스키마.
    """
    base_rule = rng.choice(space.base_rule)
    aggregation = rng.choice(space.aggregation)
    cfg: dict[str, Any] = {
        "base_rule": base_rule,
        "aggregation": aggregation,
        "segmentation": rng.choice(space.segmentation),
        "init_mode": rng.choice(space.init_mode),
        "segment_len": rng.choice(space.segment_len),
        "d_model": rng.choice(space.d_model),
        "n_layers": rng.choice(space.n_layers),
        "n_heads": rng.choice(space.n_heads),
    }
    # top_k는 ssc(aggregation) 전용 — 그 외에는 무의미하므로 생략
    if aggregation == "ssc":
        cfg["top_k"] = rng.choice(space.top_k)

    if space.ssl:
        cfg["ssl"] = {
            "encoder": rng.choice(space.ssl.get("encoder", ENCODER)),
            "invariance_coeff": rng.choice(
                space.ssl.get("invariance_coeff", INVARIANCE_COEFF)
            ),
            "positive_pair": rng.choice(
                space.ssl.get("positive_pair", POSITIVE_PAIR)
            ),
        }
        _theory_prune_ssl(cfg, rng)
    return cfg


def config_hash(cfg: dict[str, Any]) -> str:
    """config의 안정적 해시 — 중복 샘플 검출/리더보드 키 보조용."""
    blob = json.dumps(cfg, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]
