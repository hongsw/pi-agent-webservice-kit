# grokking 튜닝 — 8변형 전부 회상 학습 + 아키텍처별 용량 천장

> **결론**: 8변형 모두 MQAR 연관회상을 **학습한다(>chance)**. 단 *정확회상(near-1.0)을 grok하는
> 난이도*는 아키텍처 메커니즘에 따라 크게 다르다. softmax·영속메모리·게이팅·(감쇠+용량)이 있으면
> 다중키 회상을 grok하고, **순수 선형 어텐션(감쇠·게이트·선택 전무)**은 다중키에서 천장(~0.5)에 갇힌다.

검증 환경: RTX 4090, 실학습. 도구: `grok_tune.py`. 과제: MQAR(vocab32). 지표: query 위치 recall.
grokking 가속 레버(Power et al.): weight decay · 충분한 스텝 · warmup/cosine · (선형계열) 상태 용량.

## 종합 결과 (2키 회상, chance≈0.062)

| 변형 | 최고 recall | grok 점프 | 그것을 가능케 한 것 |
|---|---|---|---|
| swla_residual | **1.00** | step 2k | softmax 어텐션 |
| titans_ssc | **1.00** | 4k | 영속메모리 + top_k 선택 |
| linear_grm | **0.93** | 5.5k | head 게이팅 |
| dla_residual | **1.00** | 8k | 데이터 의존 감쇠 + 상태 용량↑(dh=128) |
| titans_residual | **1.00** | 13k | 영속메모리(느리지만 grok) |
| linear_ssc | 0.58 | ~20k 진행형 | top_k 선택(매우 느림) |
| linear_residual | 0.54 (2키 천장) | 없음 | — (순수 선형) |
| linear_soup | 0.50 (2키 천장) | 없음 | — (순수 선형) |

### 1키 회상(자명)에서는 전부 1.0
`linear_residual`·`linear_soup`도 pairs=1에서는 **즉시 1.0**(step 1k). → 회상 자체는 학습 가능하며,
막히는 것은 *다중키* 정확회상 용량이다. 천장은 "2키 중 1키만 안정 회상"(=0.5)으로 나타난다.

### 긴 컨텍스트(seq=512, 16×)에서도 titans는 1.0 grok ⭐
titans_residual을 seq=512(16× 긴 컨텍스트), dh=128, chunked, 25k 스텝으로 학습:
**recall 1.0(hard 0.62), grok@15000** (step14k까지 0.33 정체 → 0.60→0.79→0.94→1.0 급점프).
→ 긴 컨텍스트에서도 **완벽 회상 + 추론 O(1) 상수메모리** 동시 달성. 대가는 학습 컴퓨트(LONGSEQ.md).

## 시도한 레버와 효과
| 레버 | 효과 |
|---|---|
| 쉬운 과제(pairs 8→4→2) | 약한 변형도 학습 시작(부분해 0.34→0.53) |
| 스텝 ↑ (6k→20k) | titans_residual을 13k에서 grok시킴 |
| weight decay ↑ (0.1→0.5) | 순수 선형엔 무효(천장 불변) |
| **상태 용량 ↑ (head 8→2, dh 32→128, 상태 16×)** | **dla_residual을 grok시킴**(선형 어텐션 회상 용량 ∝ 상태 크기 입증) |

## 해석 (AutoResearch 관점)
이것이 본 노드의 존재 이유를 정확히 보여준다 — **"이 데이터의 회상 난이도에 맞는 (표현→기억)
구성"을 자동 탐색**:
- 회상 요구가 크면 → softmax(swla)/영속메모리(titans)/게이팅(grm)/감쇠+큰상태(dla, head↓) 선택.
- 순수 선형(linear+residual/soup)은 효율은 최고지만 다중키 정확회상엔 부적합 → 효율↔정확도 트레이드오프.
- 이 천장은 선형 어텐션의 알려진 MQAR 한계(Zoology)와 일치. 스텝·wd로는 못 뚫고 **용량**이 유효 레버.

## 재현
```
python3 grok_tune.py --variants all --steps 8000 --wd 0.1 --pairs 2 --vocab 32 --seq-len 16
python3 grok_tune.py --variants dla_residual --steps 12000 --pairs 2 --n-heads 2   # 용량↑로 grok
python3 grok_tune.py --variants linear_residual,linear_soup --pairs 1              # 1키는 전부 1.0
```
