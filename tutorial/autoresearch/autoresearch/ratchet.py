"""karpathy/autoresearch의 핵심 원형 — ratchet(keep-or-revert).

원 컨셉: 에이전트가 train.py를 고치고 고정 예산으로 학습 → 지표(val_bpb)가 좋아지면
keep(`git commit`), 나빠지면 revert(`git reset HEAD~1`). 즉시 개선만 누적하는 '래칫'.

이 모듈은 그 의사결정 원형을 도메인 독립적으로 제공한다:
  - Ratchet : 지표 방향(낮을수록/높을수록 우수)을 알고, 새 지표를 채택할지 판단.
  - SnapshotStore : keep/revert를 파일 스냅샷으로 안전하게 구현(메인 repo git 미접촉).
  - git 모드 : karpathy 원본처럼 지정 디렉터리에서 git commit/reset 실행(opt-in).

리더보드/ASHA(일반화 레이어)는 이 원형 위에서 best 추적을 last-write-wins로 확장한다.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass, field


@dataclass
class KeepDecision:
    keep: bool
    metric: float
    best_before: float | None
    reason: str


class Ratchet:
    """지표 래칫. better='lower'면 val_bpb처럼 낮을수록 우수, 'higher'면 proxy처럼 높을수록 우수."""

    def __init__(self, better: str = "lower", min_delta: float = 0.0):
        assert better in ("lower", "higher")
        self.better = better
        self.min_delta = min_delta  # 노이즈 무시용 최소 개선폭
        self.best: float | None = None
        self.history: list[KeepDecision] = []

    def _improves(self, metric: float) -> bool:
        if self.best is None:
            return True
        if self.better == "lower":
            return metric <= self.best - self.min_delta
        return metric >= self.best + self.min_delta

    def propose(self, metric: float) -> KeepDecision:
        """새 실험 지표를 제안. 개선이면 keep(best 갱신), 아니면 revert 권고."""
        best_before = self.best
        if self._improves(metric):
            self.best = metric
            d = KeepDecision(True, metric, best_before, "improved → keep")
        else:
            d = KeepDecision(False, metric, best_before, "no improvement → revert")
        self.history.append(d)
        return d

    def stats(self) -> dict:
        kept = sum(1 for d in self.history if d.keep)
        return {
            "experiments": len(self.history),
            "kept": kept,
            "reverted": len(self.history) - kept,
            "best": self.best,
        }


class SnapshotStore:
    """keep/revert를 파일 스냅샷으로 구현(기본·안전). 메인 repo git을 건드리지 않는다.

    karpathy 원본의 `git commit`(keep) / `git reset HEAD~1`(revert)을 단일 파일 단위로
    재현한다. accept()는 현재 파일을 '확정 스냅샷'으로 저장, revert()는 그 스냅샷으로 복원.
    """

    def __init__(self, target_file: str, snapshot_dir: str | None = None):
        self.target = os.path.abspath(target_file)
        self.snapshot_dir = snapshot_dir or os.path.join(
            os.path.dirname(self.target), ".ratchet_snapshots"
        )
        os.makedirs(self.snapshot_dir, exist_ok=True)
        self._committed = os.path.join(self.snapshot_dir, "committed.snapshot")

    def init_baseline(self) -> None:
        """현재 파일을 최초 확정 스냅샷으로."""
        if os.path.exists(self.target):
            shutil.copy2(self.target, self._committed)

    def accept(self) -> None:
        """keep — 현재 파일 상태를 확정 스냅샷으로 갱신(git commit 대응)."""
        shutil.copy2(self.target, self._committed)

    def revert(self) -> None:
        """revert — 마지막 확정 스냅샷으로 파일 복원(git reset HEAD~1 대응)."""
        if os.path.exists(self._committed):
            shutil.copy2(self._committed, self.target)


# ── git 모드(opt-in) — karpathy 원본과 동일한 메커니즘 ────────────────────────
def git_commit(repo_dir: str, message: str) -> str | None:
    """repo_dir에서 변경을 커밋하고 짧은 해시 반환. 실패 시 None."""
    try:
        subprocess.run(["git", "-C", repo_dir, "add", "-A"], check=True,
                       capture_output=True)
        subprocess.run(["git", "-C", repo_dir, "commit", "-m", message], check=True,
                       capture_output=True)
        out = subprocess.run(["git", "-C", repo_dir, "rev-parse", "--short", "HEAD"],
                             check=True, capture_output=True, text=True)
        return out.stdout.strip()
    except Exception:
        return None


def git_revert_last(repo_dir: str) -> bool:
    """karpathy 원본의 revert: 마지막 커밋을 되돌림(git reset --hard HEAD~1)."""
    try:
        subprocess.run(["git", "-C", repo_dir, "reset", "--hard", "HEAD~1"],
                       check=True, capture_output=True)
        return True
    except Exception:
        return False


def current_commit(repo_dir: str = ".") -> str | None:
    """재현용 코드 커밋 해시(리더보드 code_commit 필드에 기록)."""
    try:
        out = subprocess.run(["git", "-C", repo_dir, "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True)
        return out.stdout.strip() or None
    except Exception:
        return None
