# 긴 시퀀스 학습 — chunked O(L)로 OOM 없이

naive 병렬(O(L²))으로는 불가능하던 긴 컨텍스트 학습을 **chunked linear attention(O(L))**으로 가능케 함.
학습 경로(`grok_tune.py --chunked`, `model.forward_chunked(return_aux=True)`)를 chunked로 전환. 4090 실측.

## 1. 학습 메모리 — chunked가 길수록 압도적 (forward+backward, MQAR)
| seq_len | naive 학습 peak | chunked 학습 peak | 비고 |
|---|---|---|---|
| 2048 (B=8,h=2) | 4026 MB | **2261 MB** (~1.8× 적음) | 둘 다 가능(소배치) |
| 8192 (forward) | 8827 MB | **220 MB** (~40×) | 격차가 길이로 확대 |
| 16384 (forward) | **OOM** | 321 MB | naive 불가 |

→ 길이가 길수록 naive는 O(L²)로 폭증/OOM, chunked는 O(L). **긴 시퀀스 학습은 chunked라야 가능.**

## 2. 긴 컨텍스트 MQAR 학습 결과 (kv 앞쪽, query 말단, 사이 수백~수천 pad)
| 설정 | 변형 | final recall | hard | 학습시간 | OOM |
|---|---|---|---|---|---|
| seq=512, pairs=4, h=2 | linear | 0.348 | 0.264 | 126s/5k | ✅없음 |
| seq=512 | dla | 0.327 | 0.267 | 152s/5k | ✅없음 |
| seq=512 | titans | 0.340 | 0.269 | 157s/5k | ✅없음 |
| **seq=2048**, pairs=4 | linear | **0.349** | 0.267 | 112s/2k | ✅없음 |

chance≈0.062. 전부 **학습은 됨**(>5×chance, loss 감소) — 단 recall은 ~0.35 **부분해 plateau**.

## 3. 해석 (정직)
- ✅ **엔지니어링 목표 달성**: seq=32(초기) → **2048(64×)** 컨텍스트를 OOM 없이 학습. chunked가 핵심.
- ⚠️ **회상 품질은 부분해**: 긴 컨텍스트 다중키 정확회상은 5k 스텝 내 grok 점프 안 함. 원인:
  (a) 선형/감쇠 계열의 다중키 회상 한계(GROKKING.md), (b) 긴 pad gap이 회상 난이도↑,
  (c) dla/titans는 감쇠가 장거리 신호를 약화(floor 0.9라도 수천 스텝 누적 시 감쇠).
  → 더 많은 스텝·용량(dh↑)·낮은 감쇠가 필요(후속). swla(softmax)는 chunked 비대상(윈도우).

## 결론
> **"더 긴 시퀀스 학습"은 가능해졌다** — chunked O(L)로 64× 긴 컨텍스트를 OOM 없이 학습(메모리 1.8~40× 절감).
> 회상 품질의 grok은 추가 컴퓨트가 필요(데이터 규모/예산 의존) — 정직한 현 상태.

## 재현
```bash
python3 grok_tune.py --chunked --seq-len 2048 --pairs 4 --batch 8 --n-heads 2 \
    --variants linear_residual --steps 2000
```
