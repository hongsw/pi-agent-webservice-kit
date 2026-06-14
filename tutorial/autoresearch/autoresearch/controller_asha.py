"""§4 컨트롤러 — ASHA(Asynchronous Successive Halving Algorithm).

다수 config를 *작은 예산(rung0)*으로 시작 → 각 rung에서 상위 1/eta만 *큰 예산*으로 승급.
비동기라 GPU 유휴를 최소화한다. karpathy ratchet의 '즉시 개선만 keep'을 다중 config로
일반화: 여기선 '각 rung에서 상위만 살아남아 더 큰 예산을 받는' 토너먼트형 조기중단.

이 구현은 단일 프로세스에서 job을 하나씩 내주지만(get_job/report), 로직 자체는 병렬
워커 다수가 report를 섞어 보내도 동작하도록 설계되어 있다(§6 병렬 스윕).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class Job:
    trial_id: str
    cfg: dict[str, Any]
    rung_idx: int          # 이번에 도달할 rung
    budget_steps: int      # 이 rung의 누적 학습 스텝 목표
    is_new: bool           # 신규 trial(rung0)인지 승급인지


@dataclass
class _Trial:
    trial_id: str
    cfg: dict[str, Any]
    scores: dict[int, float] = field(default_factory=dict)  # rung_idx -> proxy score
    max_rung: int = -1     # 현재까지 평가 완료된 최고 rung
    promoted_from: set[int] = field(default_factory=set)    # 승급 처리된 rung 집합


class ASHAController:
    def __init__(
        self,
        sampler: Callable[[], dict[str, Any] | None],
        rungs: list[int],
        reduction_factor: int = 4,
    ):
        """sampler: 유효성 게이트를 통과한 새 cfg를 반환(고갈 시 None).
        rungs: 각 rung의 누적 학습 스텝(오름차순). 예: [1000, 4000, 16000].
        """
        assert rungs == sorted(rungs) and len(rungs) >= 1
        self.sampler = sampler
        self.rungs = rungs
        self.eta = reduction_factor
        self.trials: dict[str, _Trial] = {}
        self._counter = 0

    # ── 작업 분배 ────────────────────────────────────────────────────────────
    def get_job(self) -> Job | None:
        """다음 실행할 job. 승급 우선(상위 rung부터), 없으면 신규 trial 샘플."""
        promo = self._find_promotable()
        if promo is not None:
            trial, to_rung = promo
            trial.promoted_from.add(to_rung - 1)
            return Job(trial.trial_id, trial.cfg, to_rung, self.rungs[to_rung], is_new=False)

        cfg = self.sampler()
        if cfg is None:
            # 더 샘플할 게 없으면, 남은 승급만 처리(promo가 None이면 종료 신호)
            return None
        self._counter += 1
        tid = f"t{self._counter:04d}"
        self.trials[tid] = _Trial(trial_id=tid, cfg=cfg)
        return Job(tid, cfg, rung_idx=0, budget_steps=self.rungs[0], is_new=True)

    def _find_promotable(self) -> tuple[_Trial, int] | None:
        """상위 rung부터, 그 rung 상위 1/eta에 들고 아직 승급 안 된 trial을 찾는다."""
        for r in reversed(range(len(self.rungs) - 1)):
            # rung r에서 점수가 있는 trial들
            at_r = [t for t in self.trials.values() if r in t.scores]
            n = len(at_r)
            k = n // self.eta  # 승급 정원
            if k < 1:
                continue
            at_r.sort(key=lambda t: t.scores[r], reverse=True)
            top = at_r[:k]
            for t in top:
                if r not in t.promoted_from and (r + 1) not in t.scores:
                    return (t, r + 1)
        return None

    def report(self, trial_id: str, rung_idx: int, score: float) -> None:
        """워커가 rung_idx에서의 proxy score를 보고."""
        t = self.trials.get(trial_id)
        if t is None:
            return
        t.scores[rung_idx] = score
        t.max_rung = max(t.max_rung, rung_idx)

    # ── 상태 조회 ────────────────────────────────────────────────────────────
    def survivors(self, rung_idx: int | None = None) -> list[_Trial]:
        """특정 rung(기본: 최상위)까지 도달한 생존 trial."""
        target = (len(self.rungs) - 1) if rung_idx is None else rung_idx
        return [t for t in self.trials.values() if target in t.scores]

    def is_done(self) -> bool:
        """샘플 고갈 + 승급 대기 없음."""
        return self.sampler.__dict__.get("_exhausted", False) and self._find_promotable() is None

    def stats(self) -> dict[str, Any]:
        finished_top = self.survivors(len(self.rungs) - 1)
        return {
            "trials": len(self.trials),
            "rungs": self.rungs,
            "eta": self.eta,
            "reached_top_rung": len(finished_top),
        }
