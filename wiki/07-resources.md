# 07. 참고문헌 및 학습 자료

> 이 키트의 핵심 개념들을 더 깊이 이해하기 위한 참고 문헌과 링크 모음입니다.
> [과제안내](../과제안내.md) · [README](../README.md)

---

## 1. Pi 플랫폼 공식 문서

| 자료 | URL | 설명 |
|------|-----|------|
| Pi 공식 문서 | <https://pi.dev/docs/latest> | Pi 플랫폼 전반 (에이전트 정의, 실행, 설정) |
| Pi Skills 문서 | <https://pi.dev/docs/latest/skills> | Skill 작성법, SKILL.md 명세, 등록 방법 |
| Pi Packages 문서 | <https://pi.dev/docs/latest/packages> | Extension 패키지 구조, manifest 작성법 |
| Agent Skills 개요 | <https://agentskills.io/home> | Skill 철학과 설계 원칙 |
| Agent Skills 명세 | <https://agentskills.io/specification> | SKILL.md 공식 스키마 |

---

## 2. MCP (Model Context Protocol)

| 자료 | URL | 설명 |
|------|-----|------|
| MCP 공식 사이트 | <https://modelcontextprotocol.io> | MCP 프로토콜 개요 및 명세 |
| MCP 명세 | <https://spec.modelcontextprotocol.io> | JSON-RPC over stdio 상세 |
| MCP 서버 예시 | <https://github.com/modelcontextprotocol/servers> | 다양한 MCP 서버 구현 예시 |

---

## 3. karpathy/autoresearch 원 컨셉

| 자료 | URL | 설명 |
|------|-----|------|
| karpathy/autoresearch | <https://github.com/karpathy/autoresearch> | AI 에이전트가 train.py를 자유 편집해 밤새 자율 실험하는 원형 |

### 핵심 개념 정리

- **val_bpb (validation bits-per-byte)**: 언어 모델 평가 지표. 낮을수록 우수.
  vocab 크기에 독립적이어서 아키텍처 변경 간 공정 비교가 가능.
  `bpb = loss / log(2)` (nat → bit 변환)

- **ratchet 루프**: 개선되면 `git commit`(keep), 나빠지면 `git reset HEAD~1`(revert).
  밤새 ~12 experiments/hour, ~100회 실험, 15~20개 개선을 유지.

- **creativity ceiling**: ratchet 방식의 한계.
  즉시 개선만 수용하기 때문에 "일시적 후퇴 후 더 큰 개선"을 탐색하지 못함.

- **이 키트에서의 극복**: ASHA + 프록시 평가로 다중 config를 병렬 탐색하고
  조기중단으로 예산 효율을 높임.

---

## 4. ASHA (Successive Halving / Hyperband)

| 자료 | URL | 설명 |
|------|-----|------|
| Successive Halving 원논문 | <https://arxiv.org/abs/1502.07943> | Jamieson & Talwalkar (2015) |
| Hyperband 논문 | <https://arxiv.org/abs/1603.06560> | Li et al. (2017): Successive Halving의 확장 |
| ASHA 논문 | <https://arxiv.org/abs/1810.05934> | Li et al. (2018): 비동기 Successive Halving |
| Ray Tune ASHA 설명 | <https://docs.ray.io/en/latest/tune/api/schedulers.html> | 실용적 ASHA 설명과 파라미터 |

### 핵심 파라미터

| 파라미터 | 설명 | 이 키트 기본값 |
|---------|------|-------------|
| `rungs` | 각 rung의 예산(스텝 수) | [1000, 4000, 16000] |
| `eta` (η) | 각 rung에서 상위 1/η를 승급 | 3 |
| `max_t` | 최대 예산 | 16000 |

---

## 5. SSL (Self-Supervised Learning) — 표현 학습

### V-JEPA (Video Joint Embedding Predictive Architecture)

| 자료 | URL | 설명 |
|------|-----|------|
| V-JEPA 논문 | <https://arxiv.org/abs/2404.08471> | Assran et al. (2024): 비디오 JEP 아키텍처 |
| JEPA 개요 (LeCun) | <https://openreview.net/forum?id=BZ5a1r-kVsf> | LeCun (2022): JEPA 원형 아이디어 |
| Meta AI V-JEPA | <https://ai.meta.com/research/vjepa/> | 코드 및 사전학습 가중치 |

**이 키트에서의 역할**: `encoder=vjepa`로 설정 시 V-JEPA 인코더를 표현 축으로 사용.
시각적 패치의 시공간 표현을 잠재 공간에 예측적으로 임베딩.

### DINOv2

| 자료 | URL | 설명 |
|------|-----|------|
| DINOv2 논문 | <https://arxiv.org/abs/2304.07193> | Oquab et al. (2023): 자기지도 Vision Transformer |
| DINOv2 코드 | <https://github.com/facebookresearch/dinov2> | 사전학습 가중치 포함 |

**이 키트에서의 역할**: `encoder=dinov2`로 설정 시 DINOv2 ViT-B/14 사용.
강력한 범용 시각 표현, 다양한 downstream task에서 우수한 성능.

### VICReg

| 자료 | URL | 설명 |
|------|-----|------|
| VICReg 논문 | <https://arxiv.org/abs/2105.04906> | Bardes et al. (2022): Variance-Invariance-Covariance Regularization |
| Balestriero & LeCun | <https://arxiv.org/abs/2205.11508> | SSL의 닫힌형 해석: SSL이 커널 PCA의 일반화임을 보임 |

**이 키트에서의 역할**: `encoder=vicreg`으로 설정 시 VICReg 기반 표현 사용.
`invariance_coeff` 파라미터로 불변성 정규화 강도 조절.

**Balestriero & LeCun (2205.11508) 핵심**:
SSL 방법(SimCLR, VICReg, BYOL 등)이 닫힌형(closed-form)으로 분석 가능하며,
모두 커널 PCA의 특수 케이스임을 증명. 표현 축 설계 시 이론적 배경으로 참조.

---

## 6. growing-memory-pytorch

| 자료 | URL | 설명 |
|------|-----|------|
| growing-memory-pytorch | <https://github.com/xide-projext/growing-memory-pytorch> | 확장 가능한 메모리 시퀀스 모델 라이브러리 |

**이 키트에서의 역할**:
- `base_rule` (linear/swla/dla/titans): 메모리 갱신 규칙
- `aggregation` (residual/grm/soup/ssc): 기억 집약 방식
- `segmentation` (constant/logarithmic): 시퀀스 분할 전략
- `model_adapter.py`가 import 시도, 없으면 mock ToyModel로 폴백

---

## 7. MQAR (Multi-Query Associative Recall)

| 자료 | URL | 설명 |
|------|-----|------|
| MQAR 원논문 | <https://arxiv.org/abs/2205.14135> | Fu et al. (2022): Hungry Hungry Hippos (H3) 논문에서 소개 |
| 합성 recall 과제 설명 | <https://arxiv.org/abs/2302.06555> | Zoology 논문: MQAR를 벤치마크로 체계화 |

**이 키트에서의 역할**:
`proxy.py`의 `factory_mqar` 함수가 MQAR 합성 과제로 proxy_score를 계산.
시퀀스 메모리 구조의 associative recall 능력을 1k 스텝에서 빠르게 측정.

---

## 8. 시퀀스 메모리 아키텍처 배경

| 자료 | URL | 설명 |
|------|-----|------|
| Titans 논문 | <https://arxiv.org/abs/2501.00663> | Ali et al. (2025): Titans 메모리 아키텍처 |
| GRM (Gated Recurrence with Memory) | <https://arxiv.org/abs/2406.06484> | aggregation=grm의 이론적 배경 |
| SWLA (Sliding Window Linear Attention) | 다양한 linear attention 문헌 | base_rule=swla의 배경 |

---

## 9. 추가 학습 자료

| 주제 | 자료 |
|------|------|
| Python stdlib http.server | <https://docs.python.org/3/library/http.server.html> |
| JSON Lines 형식 | <https://jsonlines.org/> |
| Git 기초 (keep/revert 이해) | <https://git-scm.com/docs> |
| Spearman 상관계수 | <https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient> |
| NAS (Neural Architecture Search) 개요 | <https://arxiv.org/abs/1808.05377> |

---

## 관련 문서

- [00. 큰그림 · 진행순서](./00-overview.md)
- [06. 시스템 구조](./06-architecture.md)
- [08. 용어집](./08-glossary.md)
