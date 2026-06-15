"""growing-memory — 설계 4축(base_rule/aggregation/segmentation/init_mode) + SSL 표현축의
PyTorch 구현. 학습은 청크 병렬 O(L), 추론은 재귀 O(1) 상태(토큰 스트리밍), 둘은 동치.

이 패키지는 두 곳에 연결된다:
  1) AutoResearch 노드: model_adapter가 `import growing_memory` 후 build()/run_equivalence_test()를
     호출 → 자체 reference 대신 이 패키지를 실물 백엔드로 사용.
  2) HuggingFace/Unsloth: `growing_memory.hf`의 PreTrainedModel 래퍼로 transformers 파이프라인 로드.

설치: pip install growing-memory-pytorch  (또는 -e packages/growing-memory)
"""

from __future__ import annotations

from .model import GrowingMemoryModel, build_real
from .trainer import RealRun, train_and_eval

__all__ = [
    "GrowingMemoryModel", "build_real", "build", "build_model",
    "RealRun", "train_and_eval", "check_config", "run_equivalence_test",
    "__version__",
]
__version__ = "0.1.0"


# ── AutoResearch 노드 연결용 API ─────────────────────────────────────────────
def build_model(cfg: dict, vocab: int = 64, max_len: int = 4096) -> GrowingMemoryModel:
    """cfg로 모델 생성. AutoResearch model_adapter / 사용자 공통 진입점."""
    return build_real(cfg, vocab, max_len)


def build(cfg: dict) -> GrowingMemoryModel:
    """model_adapter._try_import_real()의 build 시드(호환). vocab/max_len은 cfg 또는 기본값."""
    return build_real(cfg, int(cfg.get("vocab", 64)), int(cfg.get("max_len", 4096)))


def run_equivalence_test(cfg: dict, tol: float = 1e-3) -> bool:
    """§3.4 유효성/동치 게이트 — 작은 차원에서 청크 병렬 vs 재귀 출력이 동치인지.

    AutoResearch model_adapter.try_equivalence_test가 이 함수를 호출한다(있으면).
    구조 오류/NaN/비동치면 False → 무효 config를 학습 전에 차단.
    """
    import torch
    small = dict(cfg)
    small["d_model"] = min(int(cfg.get("d_model", 64)), 64)
    small["n_layers"] = min(int(cfg.get("n_layers", 2)), 2)
    # n_heads는 d_model 약수로 보정
    h = int(small.get("n_heads", 4))
    while small["d_model"] % h:
        h -= 1
    small["n_heads"] = max(1, h)
    try:
        torch.manual_seed(0)
        m = build_real(small, 32, 256).eval()
        x = torch.randint(1, 32, (2, 64))
        with torch.no_grad():
            y_par = m(x)
            y_chunk = m.forward_chunked(x)
            y_rec = m.forward_recurrent(x)
        if torch.isnan(y_par).any():
            return False
        d1 = (y_par - y_chunk).abs().max().item()
        d2 = (y_par - y_rec).abs().max().item()
        return d1 < tol and d2 < tol
    except Exception:
        return False


# check_config는 run_equivalence_test의 별칭(model_adapter가 둘 다 탐색)
check_config = run_equivalence_test


# ── HuggingFace/Unsloth 래퍼 (transformers 있을 때만) ────────────────────────
try:
    from .hf import GrowingMemoryConfig, GrowingMemoryForCausalLM  # noqa: F401
    __all__ += ["GrowingMemoryConfig", "GrowingMemoryForCausalLM"]
except Exception:  # transformers 미설치 — 코어는 그대로 동작
    pass
