"""§7 NAS 데이터 인터페이스 — 커밋된 샤드만 read, 핫샤드 로컬 캐시.

수집기가 *쓰는 중* 파일은 건드리지 않는다(매니페스트에 committed=true 인 샤드만 노출).
실물에선 NFS 마운트(nfs://nas/manifests/...) + WebDataset(tar)/mmap 바이너리를 읽고,
SSL 인코더 출력 캐시를 로컬 2TB NVMe에 둔다. 여기선 그 인터페이스만 stub로 구현한다.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Iterator


@dataclass
class Shard:
    shard_id: str
    path: str
    num_samples: int
    committed: bool = False
    embedded: bool = False  # SSL 인코더 출력 사전 임베딩 캐시 존재 여부


class NASDataInterface:
    """매니페스트 기반 샤드 접근. 커밋된 샤드만 반환, 핫샤드 로컬 캐시 경로 관리."""

    def __init__(self, manifest: str, cache_dir: str = "/tmp/autoresearch_cache"):
        self.manifest = manifest
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def list_shards(self, only_committed: bool = True) -> list[Shard]:
        """매니페스트(jsonl)에서 샤드 목록. 파일 없으면 빈 목록(루프는 합성 데이터로 진행)."""
        path = self._local_manifest_path()
        if not path or not os.path.exists(path):
            return []
        shards: list[Shard] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                s = Shard(
                    shard_id=row.get("shard_id", ""),
                    path=row.get("path", ""),
                    num_samples=int(row.get("num_samples", 0)),
                    committed=bool(row.get("committed", False)),
                    embedded=bool(row.get("embedded", False)),
                )
                if only_committed and not s.committed:
                    continue  # 쓰는 중 샤드 미접근
                shards.append(s)
        return shards

    def _local_manifest_path(self) -> str | None:
        """매니페스트를 로컬 경로로 해석.

        - http(s)://  : NAS 도커 서버에서 매니페스트를 받아 로컬 캐시에 저장(실연결).
        - nfs://      : 마운트 미설정 시 None(stub).
        - 로컬 경로   : 존재하면 그대로.
        """
        m = self.manifest
        if m.startswith("http://") or m.startswith("https://"):
            return self._fetch_http(m)
        if m.startswith("nfs://"):
            return None
        return m if os.path.exists(m) else None

    def _fetch_http(self, url: str) -> str | None:
        """NAS HTTP 서버에서 매니페스트를 받아 캐시에 저장하고 경로 반환(없으면 None)."""
        import urllib.request
        dst = os.path.join(self.cache_dir, "manifest.jsonl")
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                data = r.read()
            with open(dst, "wb") as f:
                f.write(data)
            return dst
        except Exception:
            return None

    def cache_status(self) -> dict[str, Any]:
        committed = self.list_shards(only_committed=True)
        return {
            "manifest": self.manifest,
            "cache_dir": self.cache_dir,
            "committed_shards": len(committed),
            "total_samples": sum(s.num_samples for s in committed),
            "embedded_shards": sum(1 for s in committed if s.embedded),
        }

    def iter_samples(self, limit: int | None = None) -> Iterator[dict[str, Any]]:
        """커밋된 샤드에서 샘플 스트림. 샤드 path가 http면 NAS에서 받아 읽는다."""
        import urllib.request
        count = 0
        for shard in self.list_shards(only_committed=True):
            lines: list[str] = []
            if shard.path.startswith("http://") or shard.path.startswith("https://"):
                try:
                    with urllib.request.urlopen(shard.path, timeout=5) as r:
                        lines = r.read().decode("utf-8").splitlines()
                except Exception:
                    continue
            elif os.path.exists(shard.path):
                with open(shard.path, "r", encoding="utf-8") as f:
                    lines = f.read().splitlines()
            else:
                continue
            for line in lines:
                if limit is not None and count >= limit:
                    return
                try:
                    yield json.loads(line)
                    count += 1
                except json.JSONDecodeError:
                    continue
