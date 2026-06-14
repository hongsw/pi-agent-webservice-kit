#!/usr/bin/env python3
"""run_lab.py — karpathy ratchet 루프의 무설치 재현.

루프: train.py 수정(에이전트 대역 mutator) → 고정 예산 학습 → prepare.evaluate(val_bpb)
     → 좋아지면 keep(스냅샷/커밋), 아니면 revert(직전 스냅샷 복원) → 반복.

실물 karpathy에서는 'mutator' 자리에 AI 에이전트(이 키트의 Pi AutoResearch 에이전트)가
program.md를 읽고 train.py를 자유 편집한다. 여기서는 재현성을 위해 단순 knob 섭동을 쓴다.

사용:
    python3 run_lab.py --iterations 30 --seed 0
    python3 run_lab.py --iterations 30 --git     # git commit/reset 모드(이 디렉터리 기준)
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import random
import re
import subprocess
import sys

LAB_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.dirname(LAB_DIR)  # tutorial/autoresearch (autoresearch 패키지 포함)
sys.path.insert(0, PKG_ROOT)
sys.path.insert(0, LAB_DIR)

from autoresearch.ratchet import Ratchet, SnapshotStore, git_commit, git_revert_last  # noqa: E402
import prepare  # noqa: E402  (불변 평가 모듈)

TRAIN_FILE = os.path.join(LAB_DIR, "train.py")
_MODEL_RE = re.compile(r"^MODEL\s*=\s*(\{.*\})\s*$", re.MULTILINE)

# knob 경계 (mutator가 벗어나지 않게)
_BOUNDS = {
    "width": (64, 1024, "int"),
    "depth": (1, 24, "int"),
    "dropout": (0.0, 0.6, "float"),
    "lr": (1e-5, 1e-1, "log"),
}


def read_model() -> dict:
    text = open(TRAIN_FILE, "r", encoding="utf-8").read()
    m = _MODEL_RE.search(text)
    if not m:
        raise RuntimeError("train.py의 EDITABLE MODEL 라인을 찾지 못했습니다")
    return ast.literal_eval(m.group(1))


def write_model(model: dict) -> None:
    text = open(TRAIN_FILE, "r", encoding="utf-8").read()
    new = _MODEL_RE.sub(f"MODEL = {model!r}", text, count=1)
    open(TRAIN_FILE, "w", encoding="utf-8").write(new)


def mutate(model: dict, rng: random.Random) -> dict:
    """에이전트 대역: knob 하나를 경계 안에서 섭동(한 번에 하나만 — program.md 규칙)."""
    model = dict(model)
    key = rng.choice(list(_BOUNDS))
    lo, hi, kind = _BOUNDS[key]
    cur = float(model.get(key, lo))
    if kind == "int":
        step = rng.choice([-4, -2, -1, 1, 2, 4]) * max(1, int((hi - lo) * 0.05))
        model[key] = int(min(hi, max(lo, cur + step)))
    elif kind == "float":
        model[key] = round(min(hi, max(lo, cur + rng.uniform(-0.1, 0.1))), 3)
    else:  # log
        import math
        logv = math.log10(max(1e-9, cur)) + rng.uniform(-0.5, 0.5)
        model[key] = float(min(hi, max(lo, 10 ** logv)))
    return model


def run_train(steps: int) -> dict:
    """현재 train.py를 서브프로세스로 실행해 학습된 모델(JSON)을 받는다."""
    out = subprocess.run([sys.executable, TRAIN_FILE], cwd=LAB_DIR,
                         capture_output=True, text=True, check=True)
    return json.loads(out.stdout.strip().splitlines()[-1])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="karpathy ratchet 루프(mock)")
    ap.add_argument("--iterations", type=int, default=30)
    ap.add_argument("--steps", type=int, default=1000, help="고정 예산(원본의 5분 대응)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--git", action="store_true", help="git commit/reset 모드")
    args = ap.parse_args(argv)
    rng = random.Random(args.seed)

    snap = SnapshotStore(TRAIN_FILE)
    snap.init_baseline()
    ratchet = Ratchet(better="lower", min_delta=0.0005)  # val_bpb는 낮을수록 우수

    # 베이스라인
    model = run_train(args.steps)
    base_bpb = prepare.evaluate(model)
    ratchet.propose(base_bpb)
    print(f"baseline      val_bpb={base_bpb:.4f}  {model}")

    for i in range(1, args.iterations + 1):
        before = read_model()
        write_model(mutate(before, rng))      # 에이전트가 train.py 편집
        trained = run_train(args.steps)        # 고정 예산 학습
        bpb = prepare.evaluate(trained)        # 불변 평가
        decision = ratchet.propose(bpb)        # ratchet 판단
        if decision.keep:
            if args.git:
                h = git_commit(LAB_DIR, f"exp{i}: val_bpb {bpb:.4f}")
                tag = f"git {h}" if h else "git(skip)"
            else:
                snap.accept()
                tag = "snapshot"
            print(f"exp{i:>3}  val_bpb={bpb:.4f}  KEEP   ({tag})  best={ratchet.best:.4f}")
        else:
            if args.git:
                git_revert_last(LAB_DIR)
            else:
                snap.revert()                  # train.py 직전 스냅샷 복원
            print(f"exp{i:>3}  val_bpb={bpb:.4f}  revert        best={ratchet.best:.4f}")

    s = ratchet.stats()
    print("\n=== ratchet stats ===")
    print(json.dumps(s, ensure_ascii=False, indent=2))
    print(f"final train.py MODEL = {read_model()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
