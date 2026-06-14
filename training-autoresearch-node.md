# Training / AutoResearch 노드 — 설계 문서

**범위:** 온프레미스 제조 AI 시스템에서 **학습 + 자동 탐색(autoresearch)** 만 다룬다. 수집(엣지)·NAS 하드웨어·추론 서빙은 인터페이스 수준에서만 참조한다.
**한 줄 정의:** 단일 GPU 노드에서, `growing-memory-pytorch`의 설계공간 + SSL 표현 축을 자동으로 스윕해, *이 공장 데이터에 맞는 최적 (표현→기억) 구성*을 찾아내는 노드.

---

## 1. 시스템 내 위치

```
[Edge] ──샤드(append)──► [NAS] ──샤드(read)──► [THIS: Training/AutoResearch]
                            ▲                          │
                            └──리더보드/체크포인트──────┘
                                                       │ best config export
                                                       ▼
                                                  [추론/엣지 배포]
```

학습 노드는 NAS의 *커밋된 샤드만* 읽고, 실험 결과(리더보드·체크포인트)를 NAS에 쓴다. 수집과 완전 비동기.

---

## 2. 하드웨어 사양

| 부품 | 사양 | 근거 |
|---|---|---|
| GPU | RTX 4090 24GB (확장 시 5090 32GB) | 300M 학습 ≈ 7–8GB → 동시 2–3 config |
| CPU | 12–16코어 (Ryzen 9 / i9) | dataloader 워커 + 컨트롤러 |
| 시스템 RAM | 64GB (128GB 권장) | 병렬 런 + 프리페치 |
| 로컬 스토리지 | 2TB NVMe Gen4 | NAS 핫샤드 캐시 / 스크래치 |
| 네트워크 | 10GbE | NAS 순차 read |
| PSU / 환경 | 1000W+, 24/7 상시, 우수 에어플로우 | 4090 575W + 헤드룸 |

소프트웨어: PyTorch 2.x + 풀 CUDA/Triton(효율 커널 그대로) + `growing-memory-pytorch` + 컨트롤러(자체 ASHA 또는 Optuna/Ray Tune).

---

## 3. 탐색 공간 (autoresearch의 핵심)

스윕은 **두 축의 곱 + 소형 하이퍼파라미터**로 구성된다.

### 3.1 시퀀스 축 (growing-memory 4축)
| 파라미터 | 후보 |
|---|---|
| `base_rule` | linear / swla / dla / titans |
| `aggregation` | residual / grm / soup / ssc |
| `segmentation` | constant / logarithmic |
| `init_mode` | checkpoint / independent |
| `segment_len` | {64, 128, 256, 512} |
| `top_k` (ssc 전용) | {2, 4, 8} |

### 3.2 표현 축 (SSL)
| 파라미터 | 후보 | 비고 |
|---|---|---|
| `encoder` | JEPA(V/I-JEPA) / DINOv2 / VICReg | 실전 인코더 |
| `invariance_coeff` | low / high | 저데이터·정렬 여부로 결정(이론 가이드) |
| `positive_pair` | 연속프레임 / 동일유닛-다른시점 / 동일결함클래스 | **가장 큰 레버** |

### 3.3 소형 하이퍼파라미터
`d_model` {512,768,1024} · `n_layers` {6,12,24} · `n_heads` · `lr` · batch.

### 3.4 탐색 공간 가지치기
- **유효성 게이트:** 샘플한 config는 학습 전에 `growing-memory-pytorch`의 해당 동치/shape 테스트를 *작은 차원으로* 먼저 통과시킨다 → 논문 충실성·유효성 보장된 점만 학습에 진입.
- **이론 가지치기:** SSL 축은 2205.11508(Balestriero·LeCun)의 closed-form 가이드로 사전 축소(예: pairwise 어긋남↔낮은 invariance VICReg, 저데이터↔높은 invariance). 무작정 전수 탐색하지 않는다.

---

## 4. AutoResearch 루프

```
while budget remains:
    cfg        = controller.sample(search_space)      # ASHA/Hyperband
    if not validity_gate(cfg):  continue              # 동치/shape 테스트
    model      = build(cfg)                            # growing-memory-pytorch
    score      = train_and_eval(model, proxy_task,     # 짧은 예산
                                resource=rung)
    controller.report(cfg, score)                      # 조기중단/승급
    leaderboard.write(NAS, run_id, cfg, score)         # last-write-wins
best = leaderboard.top(NAS)
export(best) -> 추론/엣지 배포
```

- **컨트롤러:** ASHA(Asynchronous Successive Halving). 다수 config를 *작은 예산*으로 시작 → 상위만 *큰 예산*으로 승급. 비동기라 GPU 유휴 최소.
- **자원(rung) 단위:** 학습 스텝/토큰 수. 예: 1k → 4k → 16k 스텝 3단.
- **KPI:** 일일 처리 config 수(= 탐색 속도). 작은 모델 + 효율 RNN이라 싸게 많이 시도하는 게 본 시스템의 본업.

---

## 5. 프록시 과제 (조기평가용)

전체 downstream 평가는 비싸므로, 초기 rung은 **빠른 프록시**로 거른다.

- **Recall 프록시:** 공장 데이터로 만든 MQAR식 과제 — "특정 유닛/이벤트를 긴 컨텍스트 뒤에서 정확히 회상하는가". growing-memory의 핵심 가설(유효 메모리 성장→recall) 직접 측정.
- **단기 예측 프록시:** 다음 스텝/짧은 horizon 예측 오차(자기지도). 라벨 불필요.
- 상위 승급 config만 실제 과제(결함 탐지/이상 탐지) full 평가.

프록시는 빠르고 best와 상관이 높아야 한다 — 주기적으로 full 평가와의 순위 상관을 점검해 프록시를 보정한다.

---

## 6. 메모리 예산 & 병렬 스윕 (24GB 기준)

| 모델 | 정적(param+grad+AdamW) | 활성값(청크256·배치8) | 합계 | 24GB 동시 |
|---|---|---|---|---|
| 300M | ~5 GB | ~1–2 GB | ~7–8 GB | **2–3 config** |
| 500M | ~8 GB | ~2 GB | ~10 GB | 2 config |
| 1.3B | 8-bit opt + ckpt 필요 | — | ~28 GB | 1 config (5090 권장) |

운영 모드 2가지: **병렬**(여러 config 동시, 각 느림 — ASHA의 다수 저예산 런에 유리) vs **순차**(하나씩, 각 빠름). ASHA는 대체로 병렬 쪽이 처리량 유리.

---

## 7. NAS 데이터 인터페이스

- **읽기:** NAS 매니페스트에 *커밋된 샤드만* read(수집기가 쓰는 중 파일 미접근). 핫샤드는 로컬 2TB NVMe에 캐시 → NAS 의존·대역폭 병목 완화.
- **포맷:** WebDataset(tar) 또는 mmap 바이너리, 사전 토크나이즈/사전 임베딩(SSL 인코더 출력 캐시 포함).
- **쓰기:** run별 체크포인트 + 리더보드(run_id 키, last-write-wins) 기록. 실험 재현용으로 cfg·seed·코드 커밋 해시 동반 기록.

---

## 8. 산출물 & 핸드오프

- **리더보드:** (cfg, proxy_score, full_score, cost) 누적 — `docs/reproduction.md`와 동기.
- **best config export:** 가중치 + cfg + 인코더 → 추론 노드/엣지로 배포. 엣지는 고정상태 추론(길이 무관 평평한 메모리).
- **회귀 방지:** best 갱신 시 동치성 테스트 + full 평가 재확인.

---

## 9. 실행 설정 예 (config 스키마)

```yaml
run_id: 2026-06-14_ssc_titans_001
search:
  controller: asha
  rungs: [1000, 4000, 16000]      # 학습 스텝
  max_parallel: 3
proxy:
  task: factory_mqar              # | short_horizon_pred
  eval_every: 1000
space:
  base_rule:    [linear, swla, dla, titans]
  aggregation:  [residual, grm, soup, ssc]
  segmentation: [constant, logarithmic]
  init_mode:    [checkpoint, independent]
  segment_len:  [128, 256, 512]
  ssl:
    encoder:          [vjepa, dinov2, vicreg]
    invariance_coeff: [low, high]
    positive_pair:    [consecutive_frame, same_unit_multiview, same_defect_class]
data:
  manifest: nfs://nas/manifests/factory_v1.jsonl
  cache:    /mnt/nvme/shards
gate:
  equivalence_tests: true         # 학습 전 유효성 검증
```

---

## 10. 단계적 도입 (이 노드 한정)

| 단계 | 내용 |
|---|---|
| T0 | 4090 1대 + NAS 마운트. 시퀀스 축만(SSL 고정) 스윕, 프록시=합성 recall로 루프 폐쇄 검증 |
| T1 | SSL 축 추가(인코더·invariance·pair), 이론 가지치기 적용. 프록시↔full 상관 보정 |
| T2 | 모델 1.3B급/스윕 처리량 부족 시 5090 또는 GPU 2장. 추론 서빙 노드 분리 |

---

## 11. 열린 결정 (정해야 할 것)

- 프록시 과제의 구체 정의(어떤 유닛/이벤트를 recall 타깃으로?) — 공장 데이터 구조에 의존.
- SSL 인코더 1차 선택(V-JEPA vs DINOv2) — 결함 유형·이미지 특성에 의존.
- 병렬 vs 순차 스윕 기본값 — 초기 벤치로 처리량 측정 후 결정.
- 학습 목적함수(next-step / masked recon / 예측오차 이상탐지) 우선순위.
