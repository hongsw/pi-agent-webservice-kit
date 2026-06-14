# 08. 용어집

> AutoResearch 노드 시나리오에서 사용하는 핵심 용어를 정리합니다.
> 알파벳/한글 혼합 순 정렬.
> [과제안내](../과제안내.md) · [README](../README.md)

---

## A

### ASHA (Async Successive Halving Algorithm)
비동기 Successive Halving 알고리즘. 다수의 저예산 trial에서 시작해,
성능이 좋은 것만 더 높은 예산(rung)으로 승급시키는 조기중단 방식의
하이퍼파라미터 탐색 알고리즘.

- 핵심 파라미터: `rungs`, `η (eta)`, `max_t`
- 이 키트: rungs = [1000, 4000, 16000], eta = 3
- 관련 파일: `tutorial/autoresearch/autoresearch/controller_asha.py`
- 원논문: Li et al. (2018) [arXiv:1810.05934]

---

### aggregation
growing-memory 모델에서 시퀀스 메모리의 **기억 집약 방식**.

| 값 | 설명 |
|----|------|
| `residual` | 잔차(residual) 연결로 이전 메모리에 현재 입력을 더함 |
| `grm` | Gated Recurrence with Memory — 게이팅으로 정보 흐름 제어 |
| `soup` | 여러 메모리 상태를 평균(soup)해 집약 |
| `ssc` | Selective State Compression — 중요도 기반 선택적 압축 |

---

## B

### base_rule
growing-memory 모델에서 **메모리 갱신 규칙**. 각 스텝에서 메모리가 어떻게 업데이트되는지를 정의.

| 값 | 설명 |
|----|------|
| `linear` | 선형 RNN 방식 메모리 갱신 |
| `swla` | Sliding Window Linear Attention |
| `dla` | Delta Linear Attention — delta 규칙 기반 |
| `titans` | Titans 논문 방식 메모리 갱신 |

---

## C

### code_commit
trial 학습 시 사용된 코드(train.py 또는 growing-memory 버전)의 **git 커밋 해시**.
리더보드 레코드에 기록되어, 어떤 코드 버전으로 결과를 얻었는지 추적 가능.
karpathy ratchet의 `git commit` 동작에 대응.

### creativity ceiling
karpathy autoresearch 원형의 한계. ratchet 방식(즉시 개선만 keep)이기 때문에
"일시적 후퇴 후 더 큰 개선"을 탐색하지 못하는 현상.
이 키트는 ASHA + 프록시로 다중 탐색해 극복.

---

## D

### DINOv2
Meta AI의 자기지도(SSL) Vision Transformer 인코더.
강력한 범용 시각 표현을 학습. `encoder=dinov2` 설정 시 사용.
ViT-B/14 아키텍처 기반.

---

## E

### encoder
표현 축의 핵심 파라미터. 공장 비디오/이미지 데이터를 잠재 공간으로 임베딩하는 SSL 인코더.

| 값 | 방식 | 특징 |
|----|------|------|
| `vjepa` | V-JEPA | 예측적 JEP, 시공간 마스킹 |
| `dinov2` | DINOv2 | 자기지도 ViT, 범용 표현 |
| `vicreg` | VICReg | 분산-불변-공분산 정규화 |

---

## F

### factory_mqar
이 키트의 주 프록시(proxy) 평가 함수.
**MQAR**(Multi-Query Associative Recall) 합성 과제로 시퀀스 메모리의 recall 능력을 측정.
1,000 스텝만으로 proxy_score를 계산해 ASHA 조기중단에 활용.

### full_score
1k→4k→16k 스텝 전체 학습 후 실제 평가 데이터(공장 데이터)에서 얻은 점수.
proxy_score와 달리 비용이 높지만 최종 성능을 정확히 반영.
Rung 2 승급 trial에서만 계산.

---

## I

### init_mode
growing-memory 모델 초기화 방식.

| 값 | 설명 |
|----|------|
| `checkpoint` | 사전학습 체크포인트에서 초기화 |
| `independent` | 랜덤 초기화 (독립 학습) |

### invariance_coeff
VICReg에서 **불변성(invariance) 정규화 강도** 계수.
augmented view 간 표현의 유사도를 강제하는 손실 항의 가중치.
0에 가까울수록 불변성 제약이 약함, 1에 가까울수록 강함.

---

## J

### JEPA (Joint Embedding Predictive Architecture)
Yann LeCun이 제안한 자기지도 학습 아키텍처.
입력의 두 뷰를 같은 잠재 공간에 임베딩하고,
한 뷰에서 다른 뷰의 표현을 **예측**하도록 학습.
픽셀 수준 재구성 없이 추상적 표현을 학습하는 것이 특징.
V-JEPA는 비디오에 적용한 버전.

---

## L

### last-write-wins
리더보드 JSONL의 **쓰기 정책**.
동일한 `(run_id, trial_id)` 키를 가진 레코드가 여러 줄 있을 경우,
**가장 마지막으로 기록된 줄**이 유효한 레코드로 취급됨.
trial 상태 갱신(running → done)을 단순하게 구현하는 방식.

### leaderboard (리더보드)
AutoResearch 스윕의 전체 trial 결과를 기록하는 JSONL 파일.

주요 필드:

| 필드 | 타입 | 설명 |
|------|------|------|
| `run_id` | str | 스윕 실행 ID |
| `trial_id` | str | 개별 trial ID |
| `cfg` | dict | trial config (시퀀스+표현 파라미터) |
| `proxy_score` | float | 프록시 평가 점수 (0~1, 높을수록 우수) |
| `full_score` | float? | full 평가 점수 (Rung 2 이상) |
| `cost` | int | 사용된 스텝 수 |
| `rung` | int | 현재 rung (0, 1, 2) |
| `status` | str | "running" / "done" / "pruned" |
| `code_commit` | str? | 코드 git 커밋 해시 |
| `seed` | int? | 랜덤 시드 |
| `ts` | str | 기록 시각 (ISO 8601) |

---

## M

### MCP (Model Context Protocol)
에이전트와 외부 시스템(파일, DB, API) 간의 표준 인터페이스 프로토콜.
JSON-RPC over stdio 방식으로 통신.
이 키트: `autoresearch-mcp` (`web/mcp/autoresearch_mcp.py`)

### MQAR (Multi-Query Associative Recall)
시퀀스 메모리 평가용 합성 과제.
(key, value) 쌍을 제시한 후 나중에 key를 보여주고 value를 맞추는 recall 테스트.
H3 논문(2022)에서 도입, Zoology 논문(2023)에서 벤치마크로 체계화.

---

## N

### NAS (Network/NAS shard)
이 키트에서 NAS는 두 가지 의미로 사용됨:
1. **NAS (Network Attached Storage)**: 공장 데이터 샤드가 저장된 공유 스토리지
2. **NAS (Neural Architecture Search)**: 모델 구조 탐색 (일반 용어)

### NAS shard (데이터 샤드)
Edge 노드가 공유 스토리지에 커밋한 공장 데이터의 단위 파일 (`.pt` 형식).
`data_interface.py`가 샤드 목록을 조회해 학습 데이터로 로드.

### NAS manifest (샤드 매니페스트)
NAS에 커밋된 샤드 목록과 메타데이터를 기록한 파일.
`autoresearch-mcp`의 `nas_list_shards` 도구가 이를 파싱해 반환.

---

## P

### positive_pair
VICReg 학습에서 **양성 쌍(positive pair)** 생성 방식.
같은 인스턴스의 두 뷰를 같은 공간에 임베딩하도록 학습할 때 두 뷰를 어떻게 만드는지.

| 값 | 설명 |
|----|------|
| `augment` | 동일 프레임의 데이터 증강 (크롭, 컬러지터 등) |
| `temporal` | 같은 시퀀스의 시간적으로 가까운 두 프레임 |

### proxy task (프록시 과제)
full 학습 없이 저예산에서 최종 성능을 추정하기 위한 대리(proxy) 평가.
이 키트의 프록시: `factory_mqar`, `short_horizon_pred`.

### proxy_score
프록시 과제로 계산된 점수 (0~1, 높을수록 우수).
ASHA 조기중단 및 리더보드 순위에 사용.

### proxy_trust
`leaderboard-analysis` Skill이 계산하는 프록시 신뢰도.
proxy↔full Spearman ρ 기준: ρ≥0.7="high", 0.5≤ρ<0.7="medium", ρ<0.5="low".

---

## R

### ratchet (래칫)
karpathy autoresearch 원형의 핵심 루프.
실험 결과가 개선되면 keep(git commit), 나빠지면 revert(git reset HEAD~1).
이 키트에서는 `ratchet.py`가 best 갱신 시 code_commit 해시를 기록하는 방식으로 구현.

### rung
ASHA에서 **예산 단계**. 낮은 rung은 작은 예산, 높은 rung은 큰 예산.

| rung | 스텝 수 | 의미 |
|------|---------|------|
| 0 | 1,000 | 빠른 사전 필터링 |
| 1 | 4,000 | 중간 평가 |
| 2 | 16,000 | 정밀 full 평가 |

---

## S

### segmentation
growing-memory 모델에서 **시퀀스 분할 전략**.

| 값 | 설명 |
|----|------|
| `constant` | 고정 길이(segment_len)로 균등 분할 |
| `logarithmic` | 로그 스케일로 가변 길이 분할 (앞부분 짧게, 뒤로 갈수록 길게) |

### segment_len
시퀀스 분할 시 기본 세그먼트 길이. 32, 64, 128, 256 중 선택.

### short_horizon_pred
이 키트의 보조 프록시 함수. 짧은 시퀀스에서 다음 토큰 예측 손실(자기지도).
인코더 표현 품질과 메모리 통합 능력을 간접 측정.

### SSL (Self-Supervised Learning, 자기지도 학습)
레이블 없이 데이터 자체의 구조를 이용해 표현을 학습하는 방법.
이 키트의 표현 축: V-JEPA, DINOv2, VICReg 모두 SSL 기반.

### Skill
Pi 에이전트가 특정 작업을 정확하고 반복 가능하게 수행하도록 돕는
지식·절차·스크립트·리소스 묶음. `SKILL.md`로 선언.
이 키트: `validity-gate`, `leaderboard-analysis`

---

## T

### Titans
시퀀스 메모리 아키텍처의 한 종류. `base_rule=titans` 설정 시 사용.
Ali et al. (2025) 논문에서 제안.

### top_k
growing-memory에서 각 스텝에서 상위 k개의 메모리 슬롯만 갱신하는 희소 갱신 파라미터.
8, 16, 32 중 선택.

### trial
AutoResearch 스윕의 단일 실험 단위.
하나의 config(파라미터 조합)로 학습을 실행하고 점수를 얻는 것.
`trial_id`로 식별.

---

## V

### val_bpb (validation bits-per-byte)
karpathy autoresearch 원형의 평가 지표.
언어 모델이 검증 데이터를 압축하는 효율을 bits-per-byte로 측정.
낮을수록 우수. vocab 크기와 무관해 아키텍처 간 공정 비교 가능.
`bpb = cross_entropy_loss / log(2)` (nat → bit 변환)

### validity gate (유효성 게이트)
학습 시작 전 config의 동치·shape 검증으로 무효 config를 차단하는 Skill.
`skills/validity-gate/SKILL.md`

### VICReg (Variance-Invariance-Covariance Regularization)
SSL 표현 학습 방법. 세 가지 정규화 항(분산, 불변성, 공분산)으로
표현 공간의 붕괴를 방지하면서 불변성을 학습.
`encoder=vicreg` 설정 시 사용.

### V-JEPA (Video Joint Embedding Predictive Architecture)
Meta AI의 비디오 자기지도 학습 방법.
비디오의 시공간 마스킹 패치를 잠재 공간에서 예측.
`encoder=vjepa` 설정 시 사용.

---

## 관련 문서

- [00. 큰그림 · 진행순서](./00-overview.md)
- [02. Skill 설계](./02-skills.md)
- [03. MCP 연결](./03-mcp.md)
- [06. 시스템 구조](./06-architecture.md)
- [07. 참고문헌](./07-resources.md)
