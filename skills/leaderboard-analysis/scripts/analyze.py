#!/usr/bin/env python3
"""leaderboard-analysis Skill 실행 스크립트.

리더보드를 분석해 best 선정 + 프록시↔full 순위상관(proxy_trust) 판정 + 추천을 출력.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
PKG_ROOT = os.path.join(REPO, "tutorial", "autoresearch")
sys.path.insert(0, PKG_ROOT)

from autoresearch.leaderboard import Leaderboard          # noqa: E402
from autoresearch.proxy import rank_correlation           # noqa: E402
from autoresearch.validity_gate import validity_gate      # noqa: E402


def _trust(rho: float | None) -> str:
    if rho is None:
        return "unknown"
    if rho >= 0.7:
        return "high"
    if rho >= 0.4:
        return "medium"
    return "low"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="leaderboard-analysis Skill")
    ap.add_argument("--leaderboard",
                    default=os.path.join(PKG_ROOT, "runs", "leaderboard.jsonl"))
    ap.add_argument("-n", type=int, default=10)
    args = ap.parse_args(argv)

    lb = Leaderboard(args.leaderboard)
    recs = lb.latest()
    # 최상위 rung 도달(=full_score 있는) trial로 상관 계산
    scored = [r for r in recs if r.full_score is not None and r.proxy_score is not None]
    rho = None
    if len(scored) >= 2:
        rho = round(rank_correlation([r.proxy_score for r in scored],
                                     [r.full_score for r in scored]), 3)

    top = lb.top(args.n)
    best = top[0] if top else None
    recommendation = None
    if best:
        gate = validity_gate(best.cfg)
        recommendation = {
            "trial_id": best.trial_id,
            "cfg": best.cfg,
            "proxy_score": best.proxy_score,
            "full_score": best.full_score,
            "cost": best.cost,
            "backend": best.backend,
            "code_commit": best.code_commit,
            "export_safe": gate.ok,
            "gate_reason": gate.reason,
        }

    out = {
        "summary": lb.summary(),
        "proxy_full_rank_corr": rho,
        "proxy_trust": _trust(rho),
        "top": [
            {"trial_id": r.trial_id, "proxy_score": r.proxy_score,
             "full_score": r.full_score, "cost": r.cost, "cfg": r.cfg}
            for r in top
        ],
        "recommendation": recommendation,
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
