"""titans_backend — 검증된 Titans 구현(lucidrains titans-pytorch) 연결.

논문 arXiv:2501.00663의 실제 성능을 내는 정확 구현. 자체 from-scratch(titans_exact)는 학습되지
않아(chance), 실사용 Titans는 이 백엔드를 쓴다. 설치: pip install titans-pytorch.

교차검증(MQAR vocab32/pairs2/L64, 4090): lucidrains NeuralMemory recall **0.98** vs
titans_exact(우리) 0.06(chance). → lucidrains 채택.
"""

from __future__ import annotations


def available() -> bool:
    try:
        import titans_pytorch  # noqa: F401
        return True
    except Exception:
        return False


def neural_memory(dim: int, chunk_size: int = 64, **kw):
    """lucidrains titans_pytorch.NeuralMemory 생성([B,L,d]→(retrieved,state)). 없으면 ImportError."""
    from titans_pytorch import NeuralMemory
    return NeuralMemory(dim=dim, chunk_size=chunk_size, **kw)


def mac_transformer(num_tokens: int, dim: int, depth: int = 2, segment_len: int = 128,
                    num_persist_mem_tokens: int = 4, num_longterm_mem_tokens: int = 16, **kw):
    """Memory-As-Context Transformer(논문 MAC 변형). 전체 LM 아키텍처."""
    from titans_pytorch import MemoryAsContextTransformer
    return MemoryAsContextTransformer(
        num_tokens=num_tokens, dim=dim, depth=depth, segment_len=segment_len,
        num_persist_mem_tokens=num_persist_mem_tokens,
        num_longterm_mem_tokens=num_longterm_mem_tokens, **kw)
