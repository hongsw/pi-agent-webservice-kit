"""§7/§8 리더보드 — run별 결과를 NAS(여기선 로컬 파일)에 JSONL로 누적.

last-write-wins: 같은 (run_id, trial_id) 키는 마지막 기록이 유효. append-only 로그를
다시 읽을 때 키별로 최신 레코드만 살린다(수집과 비동기 안전, 재현용 코드 커밋 해시 동반).

karpathy autoresearch의 keep/revert(git) 자리를 일반화한 것: 채택 여부는 ratchet이
판단하고, 리더보드는 모든 시도의 (cfg, proxy, full, cost)를 영구 기록한다.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Record:
    run_id: str
    trial_id: str
    cfg: dict[str, Any]
    proxy_score: float | None = None
    full_score: float | None = None
    cost: float = 0.0            # 누적 학습 스텝(=자원 소모)
    rung: int = 0                # 마지막 도달 rung 인덱스
    status: str = "pending"      # pending|running|promoted|stopped|done|failed
    backend: str = "mock"        # real|mock
    seed: int | None = None
    code_commit: str | None = None
    ts: float = field(default_factory=time.time)

    def key(self) -> str:
        return f"{self.run_id}/{self.trial_id}"


class Leaderboard:
    """JSONL append-only 리더보드. 동시 쓰기에도 안전하도록 append + 원자적 read."""

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        if not os.path.exists(path):
            open(path, "a").close()

    def write(self, rec: Record) -> None:
        """레코드 한 줄 append(last-write-wins는 read 시 적용)."""
        line = json.dumps(asdict(rec), ensure_ascii=False, sort_keys=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def _read_all(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue  # 쓰는 중 잘린 줄은 무시(다음 read에서 복구)
        return rows

    def latest(self) -> list[Record]:
        """키별 최신 레코드만(last-write-wins). ts 기준 마지막 기록 채택."""
        by_key: dict[str, dict[str, Any]] = {}
        for row in self._read_all():
            key = f"{row.get('run_id')}/{row.get('trial_id')}"
            prev = by_key.get(key)
            if prev is None or row.get("ts", 0) >= prev.get("ts", 0):
                by_key[key] = row
        recs = []
        for row in by_key.values():
            row = {k: v for k, v in row.items() if k in Record.__dataclass_fields__}
            recs.append(Record(**row))
        return recs

    def top(self, n: int = 1, by: str = "full_then_proxy") -> list[Record]:
        """상위 n개. full_score 우선, 없으면 proxy_score로 정렬."""
        recs = [r for r in self.latest() if r.status not in ("stopped", "failed")]

        def sort_key(r: Record):
            if by == "proxy":
                return (r.proxy_score or -1.0,)
            # full 우선, 동률/없음은 proxy로 보조
            return (r.full_score if r.full_score is not None else -1.0,
                    r.proxy_score if r.proxy_score is not None else -1.0)

        recs.sort(key=sort_key, reverse=True)
        return recs[:n]

    def summary(self) -> dict[str, Any]:
        recs = self.latest()
        scored = [r for r in recs if r.proxy_score is not None]
        best = self.top(1)
        return {
            "trials": len(recs),
            "scored": len(scored),
            "total_cost_steps": sum(r.cost for r in recs),
            "best": asdict(best[0]) if best else None,
        }
