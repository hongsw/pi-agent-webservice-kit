#!/usr/bin/env python3
"""validity-gate Skill 실행 스크립트.

- 단일 config 검증:  --cfg '<json>'
- 탐색공간 통과율 측정: --space <run_config.yaml> -n <샘플 수>
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
PKG_ROOT = os.path.join(REPO, "tutorial", "autoresearch")
sys.path.insert(0, PKG_ROOT)

from autoresearch.config_io import load_run_config       # noqa: E402
from autoresearch.search_space import SearchSpace, sample_config  # noqa: E402
from autoresearch.validity_gate import validity_gate      # noqa: E402


def check_single(cfg: dict) -> int:
    r = validity_gate(cfg)
    print(json.dumps({"ok": r.ok, "reason": r.reason, "cfg": cfg},
                     ensure_ascii=False, indent=2))
    return 0 if r.ok else 1


def check_space(space_path: str, n: int, seed: int) -> int:
    run_cfg = load_run_config(space_path)
    space = SearchSpace.from_dict(run_cfg.get("space", {}))
    rng = random.Random(seed)
    passed = 0
    reasons: Counter = Counter()
    for _ in range(n):
        cfg = sample_config(space, rng)
        r = validity_gate(cfg)
        if r.ok:
            passed += 1
        else:
            reasons[r.reason] += 1
    print(json.dumps({
        "sampled": n,
        "passed": passed,
        "pass_rate": round(passed / n, 3) if n else 0.0,
        "grid_size_before_prune": space.grid_size(),
        "reject_reasons": dict(reasons.most_common()),
    }, ensure_ascii=False, indent=2))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="validity-gate Skill")
    ap.add_argument("--cfg", help="단일 config(JSON 문자열)")
    ap.add_argument("--space", help="run config(yaml) 경로 — 탐색공간 샘플링")
    ap.add_argument("-n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    if args.cfg:
        return check_single(json.loads(args.cfg))
    if args.space:
        return check_space(args.space, args.n, args.seed)
    ap.error("--cfg 또는 --space 중 하나가 필요합니다")


if __name__ == "__main__":
    raise SystemExit(main())
