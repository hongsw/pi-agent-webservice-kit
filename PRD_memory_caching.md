# PRD — `memory-caching`: 논문 충실 재현 오픈소스 구현체

| 항목 | 내용 |
|---|---|
| 문서 버전 | v1.0 (draft) |
| 대상 논문 | Behrouz et al., *Memory Caching: RNNs with Growing Memory*, arXiv:2602.24281v1 (2026-02-27) |
| 프로젝트 성격 | 논문 **충실 재현(faithful reimplementation)** + 재현성(reproducibility) 검증 |
| 1차 산출물 | PyTorch 라이브러리 + 학습/평가 스크립트 + 동치성 테스트 |
| 라이선스(제안) | Apache-2.0 |
| 핵심 원칙 | 모든 구현 요구사항은 논문의 **수식 번호 / 표 / 실험 설정**에 1:1로 매핑되고, 가능한 항목은 자동 테스트로 검증한다. |

---

## 1. 배경 및 동기

Transformer는 컨텍스트 길이에 따라 메모리가 커져 recall에 강하지만 O(L²) 복잡도를 갖는다. 반대로 RNN/선형 어텐션은 고정 크기 메모리로 O(L)이지만 과거를 잊어 recall-intensive 작업에서 뒤진다. **Memory Caching(MC)** 은 RNN 메모리 상태(체크포인트)를 세그먼트 단위로 캐싱해 유효 메모리를 시퀀스 길이에 따라 키우며, 복잡도를 **O(NL)** 로 두 극단 사이에서 조절한다(N = 세그먼트 수, 1 ≤ N ≤ L).

현재 공개된 공식 레퍼런스 구현이 없어, 후속 연구·실무 적용을 위해 **수식 수준으로 정확한** 재현 구현이 필요하다.

### 1.1 목표 (Goals)
- G1. 논문의 MC 프레임워크(Eq. 4–5)와 4개 집계 방식(Residual / GRM / Memory Soup / SSC)을 수식 그대로 구현.
- G2. 4개 베이스 업데이트 규칙(Linear Attention, SWLA, DLA, Titans)에 MC를 결합(Sec. 4.3, Eq. 26–36).
- G3. 두 가지 세그먼테이션(constant-size, logarithmic/Fenwick)과 두 가지 메모리 초기화(checkpoints vs independent compressors, Sec. 3.4)를 선택 가능하게 제공.
- G4. 논문이 명시한 **동치 관계**를 단위 테스트로 보장(§6).
- G5. Table 1–5, Fig. 4–5의 실험을 재현하는 스크립트·설정을 제공하고, 허용 오차 내 수치 재현.

### 1.2 비목표 (Non-Goals)
- N1. 새로운 알고리즘·아키텍처 제안 (연구 기여 아님, 재현이 목적).
- N2. 프로덕션 추론 서버/서빙 최적화 (별도 후속 프로젝트).
- N3. 논문이 사용한 비공개 학습 데이터의 배포 (FineWeb·Long-Data-Collections 등 **공개 데이터만** 사용).
- N4. 사전학습 가중치 배포 보장 (컴퓨트 확보 시 best-effort, 핵심 인도물 아님).

### 1.3 대상 사용자
연구자(아키텍처 변형·ablation), 엔지니어(장문 효율 RNN 적용), 교육용(MC 개념 학습).

---

## 2. 범위 (Scope)

### 2.1 In-Scope
1. MC 코어: 시퀀스 세그먼테이션, 세그먼트별 메모리 압축, 마지막 상태 캐싱, 집계 함수.
2. 집계 4종: Residual(Eq. 7), GRM(Eq. 9–10), Memory Soup(Eq. 14–15), SSC(Eq. 16–17).
3. 베이스 메모리 4종: Linear Attention, SWLA(c=2, Eq. 28–29), DLA(Eq. 30–33), Titans(Eq. 34–36).
4. 세그먼테이션 2종: constant-size(O(L²/C)), logarithmic(Fenwick, O(L log L)) — Sec. 4.2, Fig. 3.
5. 메모리 초기화 2종: checkpoint 연속(`M₀^(s)=M_{L(s-1)}^(s-1)`) vs independent(`M₀^(s)` 독립) — Sec. 3.4.
6. Log-Linear++ 베이스라인(Guo et al. 2025를 GRM+log-segment로 재구성) — Sec. 4.3.
7. Post-training MC: 추론 시 세그먼트별 캐싱 + 학습 파라미터 없는 이동평균 → length extrapolation.
8. LM 래퍼 + 학습/평가 스크립트(§5) + 동치성/재현성 테스트(§6).

### 2.2 Out-of-Scope (현 버전)
- 멀티노드 분산 학습 프레임워크 자체 (HF Accelerate/torchrun 연동만).
- 커스텀 CUDA/Triton 커널의 완전 최적화 (1차는 정확성 우선, 청크 병렬은 §7).

---

## 3. 기능 요구사항 (수식 매핑)

> 표기: `M^(s)_t` = 세그먼트 s의 시점 t 메모리, `M^(i)_{L(i)}` = 세그먼트 i의 캐시된 최종 상태, `q,k,v` = 쿼리/키/값, `γ` = 게이팅 계수.

### FR-1. MC 코어 프레임워크 (Eq. 4–5)
- 입력 `x ∈ R^{L×d}`를 세그먼트 `S^(1..N)`(길이 `L^(1..N)`)로 분할.
- 각 세그먼트는 베이스 규칙 `f(·)`로 메모리 갱신: `M^(s)_t = f(M^(s)_{t-1}; k_t, v_t)`.
- 세그먼트 종료 시 최종 상태 `M^(s)_{L(s)}` 캐싱.
- 출력: `y_t = Agg({M^(1)_{L(1)},…,M^(s-1)_{L(s-1)}}; M^(s)_t; q_t)`.
- **검증:** 메모리 update는 O(L) 유지, retrieval은 토큰당 O(N) → 총 O(NL).

### FR-2. Residual Memory (Eq. 7)
- `y_t = M^(s)_t(q_t) + Σ_{i<s} M^(i)_{L(i)}(q_t)` (게이팅 없는 단순 합).
- **검증:** 선형 메모리에서는 캐시 메모리들이 사전합 가능 → 고정 크기 메모리로 collapse (Eq. 13). 동치성 테스트 EQ-1.

### FR-3. Gated Residual Memory / GRM (Eq. 9–10)
- `y_t = γ^(s)_t M^(s)_t(q_t) + Σ_{i<s} γ^(i)_t M^(i)_{L(i)}(q_t)`, `0 ≤ γ ≤ 1`.
- 게이팅: `γ^(i)_t = <u_t, MeanPooling(S^(i))>`, `u_t = x_t W_u`, softmax 정규화.
- 옵션: `MeanPooling`은 교체 가능(기본=토큰 평균). 대안 `u_t = q_t` 제공.
- **검증:** 입력 의존 γ 때문에 선형 메모리에서도 collapse 하지 않음(사전계산·재사용 불가). EQ-2.

### FR-4. Memory Soup (Eq. 14–15)
- 파라미터 자체를 데이터 의존 평균: `θ_{M*_t} = {Σ_i γ^(i)_t W^(i)_1, …, Σ_i γ^(i)_t W^(i)_c}`, `y_t = M*_t(q_t)`.
- γ 정의는 Eq. 10과 동일.
- **검증:** 선형 메모리에서 GRM과 수학적으로 동일, deep/non-linear 메모리에서 분기. EQ-3.

### FR-5. Sparse Selective Caching / SSC (Eq. 16–17)
- relevance `r^(i)_t = <u_t, MeanPooling(S^(i))>`, 여기서 `MeanPooling(S^(i)) = Σ_{j∈S^(i)} k_j`.
- 라우터: `R_t = argTop-k({r^(i)_t}_{i<s})` + 현재 online 메모리.
- `y_t = γ^(s)_t M^(s)_t(q_t) + Σ_{i∈R_t} γ^(i)_t M^(i)_{L(i)}(q_t)`.
- `MeanPooling(S^(i))`는 사전계산 → relevance·Top-k 병렬화. 선택된 메모리만 로드(학습·추론 메모리 절감).
- **검증:** k = N-1(전부 선택) → GRM과 동일. EQ-4.

### FR-6. 베이스 메모리 업데이트 규칙 (Sec. 4.3)
| 규칙 | 업데이트 | 비고 |
|---|---|---|
| Linear Attention | `M_t = M_{t-1} + v_t k_tᵀ` (Eq. 12) | matrix 메모리 |
| SWLA (c=2) | `M^(s)_t = α_t M^(s)_{t-1} + (β_t v_{t-1}k_{t-1}ᵀ + λ_t v_t k_tᵀ)` (Eq. 28) | linear → GRM=Soup |
| DLA | `M_t = M_{t-1} − η_t ∇L(M_{t-1};k,v)`, `L=−<M(k),v>` (Eq. 30) | deep MLP 메모리 |
| Titans | `M_t = α_t M_{t-1} − S_t`, `S_t = β_t S_{t-1} − η_t∇L`, `L=‖M(k)−v‖²` (Eq. 34–35) | momentum+weight decay |
- deep 메모리 기본 구조(Sec. 5, App. B): 2-layer MLP, expansion factor 4, GELU, chunk 단위 residual+layernorm: `M(x)=x+W₁σ(W₂x)`.

### FR-7. 세그먼테이션 (Sec. 4.2, Fig. 3)
- `constant`: 동일 길이 C → 비용 O(p·L²/C).
- `logarithmic`: L의 이진 표현에 따라 2의 거듭제곱 길이 세그먼트(Fenwick), 최대 N=log₂L → O(p·L·logL).
- **검증:** L=37 → 세그먼트 길이 [32,4,1] (논문 예시) 재현. SEG-1.

### FR-8. 메모리 초기화 선택 (Sec. 3.4)
- `checkpoint`: `M^(s)_0(·) = M^(s-1)_{L(s-1)}(·)` (이전 세그먼트 최종 상태에서 시작).
- `independent`: `M^(s)_0` 독립 초기화(세그먼트 간 간섭 없음).
- 설정 플래그로 노출, 두 결과의 (장단점) 비교 가능(Sec. 5.6).

### FR-9. Log-Linear++ 베이스라인 (Sec. 4.3)
- Guo et al. 2025 log-linear attention을 **GRM + logarithmic 세그먼트**로 재구성한 변형으로 구현(공정 비교용 베이스라인).

### FR-10. Post-Training MC
- 추론 시 학습 길이마다 메모리 상태 캐싱, 디코딩은 캐시 메모리의 **학습 파라미터 없는 이동평균** 사용.
- **검증:** 사전학습된 RNN에 적용 시 length extrapolation 향상(정성/정량 리포트).

### FR-11. 극단 케이스 정합 (Sec. 4.1)
- segment_len = L (N=1) → 순수 recurrent RNN으로 환원.
- segment_len = 1 + valueless vector 메모리 → gated global softmax attention 재현(Eq. 18–20).
- compressor(`q_t = 1`) + global attention 블록 = segment_len 1의 checkpoint MC와 동치(하이브리드 모델 해석).

---

## 4. 공개 API 설계 (제안)

```python
from memory_caching import MCSequenceModel, MCConfig

cfg = MCConfig(
    base_rule   = "titans",        # "linear" | "swla" | "dla" | "titans"
    aggregation = "ssc",           # "residual" | "grm" | "soup" | "ssc"
    segmentation= "constant",      # "constant" | "logarithmic"
    init_mode   = "independent",   # "checkpoint" | "independent"
    segment_len = 256,
    top_k       = 4,               # SSC 전용
    gate_input  = "u_proj",        # "u_proj" | "query"  (Eq.10 u_t 정의)
    d_model=1536, n_layers=24, n_heads=16, vocab_size=32000,
    mem_mlp_layers=2, mem_expansion=4,   # deep memory
)
model = MCSequenceModel(cfg)
logits = model(input_ids)          # (B, L, V)
```

설계 원칙: 집계·베이스규칙·세그먼테이션·초기화가 **서로 직교(orthogonal)** 하게 조합되도록 모듈 분리. 논문의 모든 (variant × rule) 조합이 설정만으로 재현 가능해야 함.

---

## 5. 실험 재현 요구사항 (Sec. 5, App. B)

### 5.1 공통 학습 설정
- 데이터: FineWeb + Long-Data-Collections 혼합.
- 컨텍스트 길이 {2K,4K,8K,16K,32K}, 세그먼트 {16,32,64,128,256,512}. 기본 LM: 4K context / 256 segment.
- 옵티마이저 AdamW, lr 4e-4, cosine annealing, batch 0.5M tokens, weight decay 0.1, vocab 32K.
- 모델 스펙(Table 6):

| 모델 | blocks | dim | heads | peak LR | tokens |
|---|---|---|---|---|---|
| 760M | 24 | 1536 | 16 | 1.25e-3 | 30B |
| 1.3B | 18 | 2048 | 8 | 7e-4 | 100B |

### 5.2 재현 대상
| ID | 산출물 | 스크립트 | 데이터셋 |
|---|---|---|---|
| RP-1 | Table 1: LM ppl + commonsense | `eval_lm.py` | Wikitext, LMB, PIQA, HellaSwag, WinoGrande, ARC-e/c, SIQA, BoolQ |
| RP-2 | Table 2: NIAH (S-NIAH-1/2/3 @4K/8K/16K) | `eval_niah.py` | passkey / number / uuid |
| RP-3 | Table 3: in-context retrieval | `eval_recall.py` | SWDE, SQuAD, FDA, TriviaQA, Drop, NQ |
| RP-4 | Table 4: LongBench | `eval_longbench.py` | LongBench 14개 task |
| RP-5 | Fig. 5: MQAR (5 seeds) | `eval_mqar.py` | MQAR |
| RP-6 | Fig. 4: training throughput | `bench_throughput.py` | — |
| RP-7 | Table 5: ablation(γ context-dep / gating / linear mem / shared u·q) | `ablation.py` | — |

### 5.3 재현 성공 기준
- S-1. 760M·30B 설정에서 Table 1 평균 정확도가 논문 대비 **±1.0%p** 이내(난수/데이터 셔플 차이 허용).
- S-2. 정성적 경향 일치: 모든 MC 변형이 동일 베이스 대비 향상; GRM ≥ SSC ≥ Log-Linear++ 경향(Sec. 5.1); 효율은 SSC가 장문에서 우위(Fig. 4).
- S-3. Table 5의 정성 결론 재현: gating·context-dependent γ 기여 양(+); **shared u=q 설정 시 학습 붕괴(정확도 ≈ 0)** 현상 재현(주의 사항으로 문서화).
- 1.3B·100B는 컴퓨트 가용 시 stretch goal.

---

## 6. 검증 / 동치성 테스트 (정확성의 핵심)

자동화 테스트로 "논문 그대로"를 보증한다.

| ID | 주장(논문 근거) | 테스트 방법 |
|---|---|---|
| EQ-1 | 선형 메모리 + Residual = 고정 크기 메모리 (Eq. 13) | 동일 입력에서 plain linear-attn RNN과 출력 일치 (atol≤1e-5) |
| EQ-2 | 선형 메모리 + GRM ≠ collapse | 입력 의존 γ가 사전합으로 환원 불가함을 반례로 확인 |
| EQ-3 | Memory Soup = GRM (linear), ≠ (deep) | linear에서 출력 일치 / deep에서 불일치 확인 |
| EQ-4 | SSC(k=N-1) = GRM | 출력 일치 (atol≤1e-5) |
| EQ-5 | N=1 → 순수 RNN | segment_len=L에서 베이스 규칙 단독과 일치 |
| EQ-6 | segment=1 + valueless = gated global attn (Eq. 20) | 수식 전개대로 gated softmax attention과 일치 |
| EQ-7 | 하이브리드(compressor+attn) = checkpoint MC(seg=1) | Sec. 4.1 단순화 버전 동치 확인 |
| SHAPE | 모든 (variant×rule×seg×init) 조합 forward/backward | 출력 shape·gradient 흐름 |
| SEG-1 | 로그 세그먼테이션 L=37 → [32,4,1] | 분할 결과 일치 |
| GRAD | 메모리 update의 inner 최적화(DLA/Titans) | 수치 미분과 해석적 gradient 일치 |

테스트는 작은 차원에서 결정론적으로 수행하고 CI에 포함한다.

---

## 7. 비기능 요구사항

- NF-1. **정확성 우선**: 1차 릴리스는 readability·동치성 우선의 reference 구현. 효율 최적화는 동치성 테스트 통과 후 적용.
- NF-2. **효율(2차)**: SSC는 선택된 메모리만 gather하여 가속기에 적재(논문 Sec. 3.3 취지). 세그먼트 내부는 청크 단위 병렬 선형 어텐션. Triton/CUDA 커널은 옵션.
- NF-3. **결정론·재현성**: seed 고정, `torch.use_deterministic_algorithms`, 설정·로그·결과 아티팩트 버전 관리.
- NF-4. **의존성 최소화**: PyTorch 2.x + 표준 라이브러리 중심. 평가 데이터셋 로더는 선택 의존성.
- NF-5. **하드웨어**: 단일 GPU에서 소형 설정 학습/테스트 가능. 760M/1.3B는 멀티 GPU 권장(가이드 제공).
- NF-6. **문서화**: 모든 변형이 논문 수식 번호와 docstring으로 연결.

---

## 8. 레포 구조 (제안)

```
memory-caching/
├─ README.md                # 개요, 논문 매핑 표, 빠른 시작
├─ pyproject.toml / LICENSE (Apache-2.0)
├─ memory_caching/
│  ├─ memory.py             # LinearAttention, SWLA, DLA, Titans (FR-6)
│  ├─ caching.py            # 세그먼테이션·캐싱·retrieval 코어 (FR-1)
│  ├─ aggregation.py        # Residual/GRM/Soup/SSC (FR-2~5)
│  ├─ segmentation.py       # constant / logarithmic (FR-7)
│  ├─ layers.py             # 블록(norm·residual), gating(Eq.10)
│  └─ model.py              # MCSequenceModel (LM 래퍼)
├─ configs/                 # 760m.yaml, 1p3b.yaml, task별 설정
├─ experiments/             # train.py, eval_*.py, bench_throughput.py, ablation.py
├─ tests/                   # test_equivalence.py(EQ-*), test_shapes, test_segmentation
└─ docs/                    # 수식↔코드 매핑, 재현 가이드
```

---

## 9. 마일스톤 / 로드맵

| 단계 | 내용 | 완료 기준 |
|---|---|---|
| M0 | MC 코어 + Linear Attention + Residual/GRM + 동치성 테스트 EQ-1,2,5 | 테스트 green |
| M1 | Soup + SSC, EQ-3,4; 세그먼테이션 2종 SEG-1 | 모든 변형 forward/backward |
| M2 | DLA·Titans·SWLA deep 메모리, GRAD 테스트 | inner-loop 수치검증 통과 |
| M3 | LM 래퍼 + train.py + RP-1(760M 소규모) | ±1.0%p 경향 일치 |
| M4 | 장문 평가 RP-2~5 + Log-Linear++ 베이스라인 | 표/그림 재현 리포트 |
| M5 | 효율화(SSC gather, 청크 병렬) RP-6 | 처리량 그래프 재현 |
| M6 | 문서·예제·v0.1.0 릴리스 | PyPI/GitHub 공개 |

---

## 10. 리스크 및 대응

| 리스크 | 영향 | 대응 |
|---|---|---|
| 논문 일부 하이퍼파라미터·정규화(γ softmax 적용 위치, MeanPooling 정의) 미명시 | 수치 불일치 | 합리적 기본값 + 설정 노출, README에 가정 명시, 저자 문의 |
| 1.3B/100B 재현 컴퓨트 비용 | 대규모 재현 어려움 | 소형(760M·축소 토큰) 우선, 대형은 stretch + 커뮤니티 기여 |
| deep 메모리 inner 최적화(Titans momentum) 불안정 | 학습 발산 | 청크 병렬 + 수치 안정화, GRAD 테스트로 회귀 방지 |
| 평가 데이터셋 라이선스/접근 | 재현 차단 | 공개 데이터만, 라이선스 표기, 비공개 항목은 스킵 옵션 |
| shared u=q 등 붕괴 설정 오용 | 재현 혼란 | 기본값에서 배제 + 경고 + Table 5로 근거 문서화 |

---

## 11. 오픈소스 운영

- 라이선스 Apache-2.0(특허 보호 포함), 논문·인용(CITATION.cff) 명시, 저자 기여 비제휴 사실 표기.
- CONTRIBUTING(테스트 통과·수식 매핑 필수), 이슈 템플릿(재현 보고), PR에 동치성 테스트 요구.
- CI: lint + 소형 동치성/shape 테스트 자동 실행.
- 재현 결과는 `docs/reproduction.md`에 표/그림과 함께 누적 기록.

---

### 부록 A. 변형 × 베이스 규칙 매트릭스 (모두 설정으로 도달 가능해야 함)

| | Linear | SWLA | DLA | Titans |
|---|---|---|---|---|
| Residual | ✔ (=collapse, EQ-1) | ✔ | ✔ | ✔ |
| GRM | ✔ | ✔ (=Soup) | ✔ | ✔ |
| Memory Soup | =GRM | =GRM | ✔(분기) | ✔(분기) |
| SSC | ✔ | ✔ | ✔ | ✔ |

### 부록 B. 핵심 수식 인덱스
Eq.4–5 코어 / Eq.7 Residual / Eq.9–10 GRM·게이팅 / Eq.13 linear collapse / Eq.14–15 Soup / Eq.16–17 SSC / Eq.18–20 valueless→gated attn / Eq.26–29 SWLA / Eq.30–33 DLA / Eq.34–36 Titans / Sec.3.4 init / Sec.4.2 세그먼테이션 / Table 1–5·Fig.4–5 재현.
