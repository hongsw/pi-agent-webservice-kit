# 비교 — 트랜스포머 vs 선형(naive/chunked 병렬) vs 선형(재귀, 메모리캐싱)

같은 차원(d256/h8/L4)에서 시퀀스 길이별 peak GPU 메모리. 4090 실측. 재현: `compare3.py`.

> 경위: 처음엔 우리 선형 '병렬'이 L×L 행렬을 materialize하는 **naive(O(L²)) 구현이라 16384에서 OOM**
> 났다. 이는 선형어텐션 장점을 버린 구현 결함 → **chunked(GLA 표준, O(L))로 재구현**해 해결.

## peak 메모리 (forward, batch=1)

| L | TF(full, FlashAttn) | 선형(naive) | **선형(chunked)** | 선형(재귀) | TF KV캐시(추론) | 재귀 상태 |
|---|---|---|---|---|---|---|
| 4096 | 182 MB | 2309 MB | **170 MB** | 170 MB | 32 MB | 132 KB |
| 8192 | 245 MB | 8827 MB | **220 MB** | 220 MB | 64 MB | 132 KB |
| 16384 | 371 MB | **OOM** | **321 MB** | — | 128 MB | 132 KB |
| 32768 | 623 MB | OOM | **522 MB** | — | 256 MB | 132 KB |

## 정확도 (동치)
naive == chunked == 재귀: 모두 같은 가중치, 출력 diff **~5e-7**(verify_recurrent.py + chunked 검증). 손실 0.
트랜스포머는 다른 아키텍처(회상은 softmax가 더 강함, GROKKING.md).

## 결론 (정정 후 — 정직)
| 항목 | 결과 |
|---|---|
| naive 병렬 OOM | ❌ 우리 구현 결함이었음(O(L²)). chunked로 해결 |
| **학습/prefill 메모리** | chunked 선형 **O(L)** — 긴 길이에선 FlashAttn 트랜스포머보다도 적음(16384: 321 vs 371MB, 32768: 522 vs 623MB) |
| **추론 메모리** | 재귀 상태 **O(1)=132KB 상수** vs 트랜스포머 KV캐시 **O(L)**(16384=128MB) → **~992×**, 길이 무관 |
| 정확도 | naive=chunked=재귀 동치(5e-7) |

→ 제대로 구현하면(chunked) 선형/Titans는 **학습 메모리는 트랜스포머와 동급 이상**,
   **추론 메모리는 압도적(상수, ~992×)**. 사용자가 지적한 OOM은 우리 naive 구현 버그였고 수정됨.

## 재현
```bash
python3 tutorial/autoresearch/compare3.py   # → compare3_result.json
```
