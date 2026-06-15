# Titans — 이름 출처, 정확 구현, 정직한 검증 결과

## 이름 출처 (제가 만든 게 아님)
- **설계 문서**(사용자 제공) §3.1의 `base_rule` 후보에 `titans`가 이미 있었음.
- 원 출처: **Google Research 논문 "Titans: Learning to Memorize at Test Time"**
  (Behrouz, Zhong, Mirrokni — arXiv:2501.00663, 2025). 어텐션 + 테스트시점 학습 신경 장기기억.

## 논문 핵심 식
```
ℓ(W; x_t) = ||W·k_t − v_t||²            (연관 손실; k=xW_K, v=xW_V, q=xW_Q)
∇ℓ        = (W·k_t − v_t)·k_tᵀ          (행렬 메모리 기울기)
S_t       = η_t·S_{t−1} − θ_t·∇ℓ         (모멘텀: 과거+현재 surprise)
W_t       = (1 − α_t)·W_{t−1} + S_t      (α_t: 데이터 의존 망각/weight-decay 게이트)
y_t       = W_t·q_t                      (현재 메모리로 읽기)
```
η_t 모멘텀, θ_t 학습률, α_t 망각(모두 데이터 의존). 추론 상태 (W,S)=O(d²) 상수 → 상수메모리 추론.

## 세 가지 "titans" 구현과 실측 (정직)
| 구현 | 무엇 | MQAR recall(vocab32/pairs2/L64, 4090) |
|---|---|---|
| `base_rule="titans"`(model.py) | dla 감쇠 + 정적 영속읽기 (근사) | 부분~1.0(grok, 학습 많이) — 단 논문 메커니즘 아님 |
| `titans_exact.py`(우리 from-scratch) | 논문 식 직접 재현 | **0.06 = chance (학습 실패)** ⚠️ |
| **lucidrains `titans-pytorch`** | 검증된 정확 구현 | **0.98** ✅ |

→ **결론: 자체 정확 구현(titans_exact)은 chance로 실패**. 안정화(키/쿼리 L2정규화+출력 norm)로
   NaN(0.0)→chance(0.06)까지 갔으나 학습은 안 됨(청크병렬·정확기울기·메모리MLP 등 미충족).
   **정확한 Titans는 lucidrains `titans-pytorch`를 채택**(`growing_memory.titans_backend`).

## 사용 (정확 구현)
```bash
pip install "growing-memory-pytorch[titans]"   # titans-pytorch 포함
```
```python
from growing_memory.titans_backend import neural_memory, available
assert available()
mem = neural_memory(dim=384, chunk_size=64)     # lucidrains NeuralMemory
retrieved, state = mem(seq)                      # [B,L,d]
```

## 교훈
사용자 지적("titans는 네가 만든 거냐?")이 정확했다. 이름은 Google 논문 차용, 자체 구현은
근사/실패였다. "정확 구현을 하거나 검색하거나" 중 **검색·채택(lucidrains)이 정답**이었음.
재현: `packages/growing-memory/titans_compare.py`.
