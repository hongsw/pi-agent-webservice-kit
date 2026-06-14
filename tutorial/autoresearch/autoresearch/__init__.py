"""AutoResearch 노드 — 단일 GPU 노드에서 (표현→기억) 구성을 자동 스윕하는 컨트롤러.

설계 문서의 §3(탐색공간) · §4(루프) · §5(프록시) · §7(NAS) · §8(산출물)을
순수 stdlib로 구현한 최소 동작 프로토타입. torch / growing-memory-pytorch가
없어도 mock 어댑터로 루프가 폐쇄(closed-loop) 검증된다.
"""

from .search_space import SearchSpace, sample_config
from .validity_gate import validity_gate, GateResult
from .leaderboard import Leaderboard, Record
from .controller_asha import ASHAController
from .loop import autoresearch_loop

__all__ = [
    "SearchSpace",
    "sample_config",
    "validity_gate",
    "GateResult",
    "Leaderboard",
    "Record",
    "ASHAController",
    "autoresearch_loop",
]
