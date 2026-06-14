"""§5 프록시 과제 — 조기 평가용 빠른 신호. full 평가는 비싸므로 초기 rung은 프록시로 거른다.

- factory_mqar : 공장 데이터 MQAR식 recall 과제(유효 메모리 성장→recall 가설 직접 측정).
- short_horizon_pred : 다음 스텝/짧은 horizon 예측오차(자기지도, 라벨 불필요).

karpathy autoresearch의 val_bpb(단일 불변 지표)에 대응하는 도메인 일반화. 모두
"높을수록 우수(0~1)"로 정규화해 컨트롤러가 동일 방향으로 비교한다.

프록시는 빠르고 best와 상관이 높아야 한다 → rank_correlation_check로 주기적 보정.
"""

from __future__ import annotations

import random
from typing import Any

from .model_adapter import ModelHandle


def factory_mqar(model: ModelHandle, rng: random.Random) -> float:
    """합성 recall 프록시 — "긴 컨텍스트 뒤에서 특정 유닛/이벤트를 회상하는가".

    mock에서는 model.memory_strength를 회상 정확도로 매핑한다(노이즈 포함).
    실물 연결 시 이 함수 안에서 합성 MQAR 시퀀스를 만들어 model로 평가하면 된다.
    """
    strength = model.memory_strength(rng)
    # 회상 난이도: 컨텍스트가 길수록 어렵다고 보고 약한 패널티(설계 직관)
    seg = int(model.cfg.get("segment_len", 256))
    difficulty = 1.0 - min(seg, 512) / 512 * 0.08
    return max(0.0, min(1.0, strength * difficulty))


def short_horizon_pred(model: ModelHandle, rng: random.Random) -> float:
    """단기 예측 프록시 — 예측오차의 역수를 0~1 점수로. 라벨 불필요(자기지도)."""
    skill = model.predict_skill(rng)
    return max(0.0, min(1.0, skill))


PROXY_TASKS = {
    "factory_mqar": factory_mqar,
    "short_horizon_pred": short_horizon_pred,
}


def evaluate_proxy(task: str, model: ModelHandle, rng: random.Random) -> float:
    """프록시 점수 — 백엔드(real/mock)별 구현은 ModelHandle.proxy_score가 담당."""
    if task not in PROXY_TASKS:
        raise ValueError(f"unknown proxy task: {task!r} (choices: {list(PROXY_TASKS)})")
    return model.proxy_score(task, rng)


def evaluate_full(model: ModelHandle, rng: random.Random) -> float:
    """실제 과제 full 평가 — 상위 승급 config에만 적용(비쌈). real=어려운 평가."""
    return model.full_score(rng)


def rank_correlation(proxy_scores: list[float], full_scores: list[float]) -> float:
    """Spearman 순위상관(근사) — 프록시가 best와 얼마나 일치하는지 보정 점검(§5).

    의존성 없이 stdlib만으로 계산. 동점은 평균 순위로 처리.
    """
    n = len(proxy_scores)
    if n < 2 or n != len(full_scores):
        return 0.0

    def ranks(xs: list[float]) -> list[float]:
        order = sorted(range(n), key=lambda i: xs[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and xs[order[j + 1]] == xs[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0  # 1-based 평균 순위
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rp, rf = ranks(proxy_scores), ranks(full_scores)
    mp = sum(rp) / n
    mf = sum(rf) / n
    cov = sum((rp[i] - mp) * (rf[i] - mf) for i in range(n))
    vp = sum((rp[i] - mp) ** 2 for i in range(n)) ** 0.5
    vf = sum((rf[i] - mf) ** 2 for i in range(n)) ** 0.5
    if vp == 0 or vf == 0:
        return 0.0
    return cov / (vp * vf)
