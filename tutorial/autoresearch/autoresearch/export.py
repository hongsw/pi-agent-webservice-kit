"""§8 best config export — 가중치 + cfg + 인코더를 추론/엣지로 핸드오프.

회귀 방지: best 갱신 시 동치성 테스트 재확인 후 export. 엣지는 고정상태 추론
(길이 무관 평평한 메모리)이므로 cfg와 인코더 식별자만으로 재구성 가능해야 한다.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

from .leaderboard import Leaderboard
from .validity_gate import validity_gate


def export_best(leaderboard_path: str, out_dir: str) -> dict[str, Any]:
    lb = Leaderboard(leaderboard_path)
    best = lb.top(1)
    if not best:
        raise RuntimeError("leaderboard is empty — run a sweep first")
    r = best[0]

    # 회귀 방지: export 전 동치성/유효성 재확인
    gate = validity_gate(r.cfg)
    if not gate.ok:
        raise RuntimeError(f"best config failed validity gate at export: {gate.reason}")

    os.makedirs(out_dir, exist_ok=True)
    bundle = {
        "exported_at": time.time(),
        "run_id": r.run_id,
        "trial_id": r.trial_id,
        "cfg": r.cfg,
        "encoder": (r.cfg.get("ssl") or {}).get("encoder"),
        "proxy_score": r.proxy_score,
        "full_score": r.full_score,
        "cost_steps": r.cost,
        "code_commit": r.code_commit,
        "seed": r.seed,
        "backend": r.backend,
        "gate": gate.reason,
        "weights": "checkpoint.pt placeholder (mock: 학습 가중치는 실물 통합 시 첨부)",
    }
    path = os.path.join(out_dir, "best_config.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(bundle, f, ensure_ascii=False, indent=2)
    return {"export_path": path, "bundle": bundle}
