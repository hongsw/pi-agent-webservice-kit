# 온프레미스 AutoResearch 토폴로지 — 구체 사양

**대상:** `growing-memory-pytorch` 기반 제조 소규모 autoresearch (수집·학습·추론을 머신 분리, NAS 공유, 온프레미스 완결)
**설계 전제:** growing-memory는 *고정 크기* 메모리로 푸는 기술 → 큰 통합메모리 박스 불필요. 학습 노드는 단일 RTX 4090(24GB)이 적정선.

---

## 0. 토폴로지 개요

```
          ┌──────────── OT / Camera VLAN (격리) ────────────┐
[CCTV ×N] ─►│ Edge Collector 1   Edge Collector 2  …  k      │
[PLC/센서]─►│  Jetson Orin NX / Mac Mini M4 (8–16GB)         │
          │   capture · preprocess · tokenize · 경량추론     │
          └───────────────────────┬─────────────────────────┘
                                   │ 2.5GbE  (append shards)
                            ┌──────▼───────┐
                            │  10GbE Switch │  (IT VLAN, OT와 분리)
                            └───┬───────┬───┘
                  10GbE (read)  │       │  10GbE (read/write)
                       ┌────────▼───┐ ┌─▼──────────────────────┐
                       │   NAS       │ │ Training / AutoResearch │
                       │ raw+shards  │◄┤  Node — RTX 4090 24GB   │
                       │ ckpt+board  │─►  controller + train     │
                       └─────────────┘ │  + local NVMe cache     │
                                       └─────────────────────────┘
        (배포: best config → Edge Collector로 push, 추론은 엣지에서)
```

핵심: **NAS가 분리 지점(decoupling point)**. 수집기는 24/7 독립적으로 샤드를 append, 학습 노드는 자기 cadence로 read. 머신은 다르지만 하드(스토리지)는 공유.

---

## 1. 노드별 메모리 적정선 (요약)

| 노드 | 역할 | 메모리 적정선 | 근거 |
|---|---|---|---|
| Edge Collector | 캡처·토크나이즈·경량추론 | 8–16 GB | 소형 모델 추론 + 고정 상태 |
| Training Node | from-scratch 학습 + config 스윕 | **24 GB VRAM** (≤500M) / 32 GB (≤1.3B) | 300M 학습 ≈ 7–8GB, 청크 활성값 |
| NAS | 공유 스토리지 | 메모리 아닌 *용량/대역폭* | §4 용량 산정 |

통합메모리 128GB 박스가 정당화되는 유일한 경우 = 남의 대형 dense VLM 추론을 겸할 때. 그건 우리 기술로 대체하려는 대상이므로 본 구성에서 제외.

---

## 2. Edge Collector (수집 노드)

| 항목 | 사양 | 비고 |
|---|---|---|
| 옵션 A | NVIDIA Jetson Orin NX 16GB | CUDA, 비전 전처리 강함, 카메라 다수 처리 |
| 옵션 B | Mac Mini M4 16GB | MLX 추론(기보유 경험 활용), 저소음·저전력 |
| 처리 | RTSP 수신 → 프레임 샘플링 → 토크나이즈/특징추출 → 경량 이상탐지 | 원천 전체 전송 금지, *토큰/특징* 위주 전송 |
| 네트워크 | 2.5GbE (최소 GbE) | OT VLAN 소속 |
| 대수 | 카메라 4–8대당 1노드 | 라인/구역 단위 |
| 로컬 버퍼 | 256GB NVMe | 네트워크 단절 시 store-and-forward |

**보안:** OT/카메라 네트워크는 IT 네트워크와 VLAN 분리. 수집기만 NAS 쓰기 경로를 가짐.

---

## 3. Training / AutoResearch Node (핵심)

| 부품 | 사양 | 근거 |
|---|---|---|
| GPU | **RTX 4090 24GB** (확장 시 5090 32GB) | 300M 학습 7–8GB → 24GB에 동시 2–3 config |
| CPU | Ryzen 9 / Core i7–i9, 12–16코어 | dataloader 워커 + 컨트롤러 |
| 시스템 RAM | **64 GB** (128 GB 권장) | 병렬 런 + 데이터 프리페치 |
| 로컬 스토리지 | 2 TB NVMe (Gen4) | NAS 핫샤드 캐시 / 스크래치 |
| 네트워크 | 10GbE NIC | NAS 순차 read |
| PSU | 1000W+ 80+ Gold | 4090 450W + 헤드룸 |
| 냉각/환경 | 우수 에어플로우, 24/7 상시 | 발열·소음·전원용량 사전 점검 |

**학습 메모리 산정 (300M, AdamW, bf16):**
파라미터+그래디언트+옵티마이저 상태 ≈ 5 GB · 청크(256)·배치(8) 활성값 ≈ 1–2 GB · **합계 ≈ 7–8 GB**.
→ 24GB에서 단일 학습은 여유, **동시 2–3 config 병렬 스윕** 가능 (= 일일 탐색량).
1.3B는 8-bit optimizer + activation checkpointing으로 32GB(5090)에서 가능.

**소프트웨어:** 풀 CUDA + Triton 그대로 → 로드맵 M5 효율 커널(SSC gather, chunked scan) 포팅 없이 작동. (통합메모리 박스 대비 이 노드를 4090으로 두는 결정적 이유.)

---

## 4. NAS (공유 스토리지 = 분리 지점)

| 항목 | 사양 | 근거 |
|---|---|---|
| 형태 | TrueNAS 빌드 or 프로슈머 NAS (Synology/QNAP xs급) | 10GbE + NVMe 캐시 가능 모델 |
| 용량 | **usable 20–40 TB** (RAID-Z2 / SHR-2) | §아래 산정 |
| 캐시 티어 | 1–2 TB NVMe (read cache / metadata) | 핫샤드 가속 |
| 네트워크 | 10GbE (다수 reader 시 LACP) | 학습 노드 순차 read |
| 프로토콜 | **NFS** (SMB 아님) | ML dataloader 친화 |

**용량 산정 예 (카메라 8대):**
원천 1080p H.265 ≈ 6 GB/cam/day → 8대 × 30일 보존 ≈ **1.4 TB rolling**.
토크나이즈 샤드는 원천의 수십 분의 1 → 수십 GB/월.
체크포인트: config당 1–5 GB × 수백 실험 → **수백 GB**.
→ 시작 usable 20 TB로 충분, 보존정책·카메라 증설에 따라 40 TB까지.

**처리량:** 10GbE 순차 read ≈ 실효 600–900 MB/s. 사전 토크나이즈 + 로컬 NVMe 캐시면 소형 모델 학습에서 dataloader 병목 거의 없음.

---

## 5. 네트워크 패브릭

| 구간 | 사양 |
|---|---|
| 코어 스위치 | 10GbE (최소 4–8 포트, 예: MikroTik CRS / QNAP / Netgear) |
| Training ↔ NAS | 10GbE 직결 또는 스위치 경유 |
| Edge ↔ 스위치 | 2.5GbE 업링크 |
| VLAN | OT(카메라/PLC) ↔ IT(NAS/학습) **분리**, 수집기만 경계 통과 |

---

## 6. 동시성 규약 (수집 write + 학습 read 충돌 방지)

- 수집기는 **append-only atomic 샤드 쓰기**: `*.tmp`로 쓰고 fsync 후 원자적 rename → 완성본만 노출.
- NAS에 **매니페스트/인덱스**(SQLite 또는 append JSONL): 커밋된 샤드만 등록.
- 학습 노드는 **매니페스트에 커밋된 샤드만** read (쓰는 중 파일 안 건드림).
- 실험 리더보드는 last-write-wins, run_id 키로 분리 기록.
- 포맷: WebDataset(tar 샤드) 또는 mmap 가능한 바이너리 → 순차 read 최적.

---

## 7. AutoResearch 루프 (토폴로지 매핑)

1. **수집기 → NAS**: 24/7 원천 + 토크나이즈 샤드 append.
2. **학습 노드**: 컨트롤러(ASHA/Hyperband)가 4축 설계공간(base rule × aggregation × segmentation × init) + 소형 하이퍼파라미터에서 config 샘플 → 캐시된 샤드로 학습 → **빠른 프록시 과제**(공장 데이터 기반 MQAR식 recall / 단기 예측)로 조기평가 → 가지치기 → 리더보드를 NAS에 기록 → 다음 config.
3. **배포**: best config export → Edge Collector로 push, 엣지에서 고정상태 추론.

동치성 테스트 하네스가 후보 config의 *유효성/논문 충실성*을 보장 → 탐색공간의 모든 점이 신뢰 가능.

---

## 8. 단계적 도입 (Phased Rollout)

| 단계 | 구성 | 목표 |
|---|---|---|
| Phase 0 (파일럿) | 학습노드 1(4090) + NAS(20TB) + Edge 1–2 | 1개 라인/카메라로 루프 폐쇄 검증 |
| Phase 1 (확장) | Edge를 전 카메라로, NVMe 캐시 티어 추가 | 데이터 처리량·보존 정책 안정화 |
| Phase 2 (증강) | 5090 또는 GPU 2장, 추론 서빙 노드 분리 | 모델 1.3B급 / 스윕 처리량 / 상시 서빙 |

---

## 9. 개략 BOM (USD, *변동성 큼*)

> 2026년 DRAM/GPU 공급난으로 가격 변동 큼. 아래는 사양 기준 *개략 범위*이며 실거래가는 별도 견적 필요.

| 품목 | 사양 | 개략가 |
|---|---|---|
| Training Node | 4090 24GB + R9/i9 + 64GB + 2TB NVMe + 1000W + 10GbE | $3,000–4,500 |
| NAS | 6-bay, 20TB usable, 10GbE, NVMe 캐시 | $2,000–3,500 |
| Edge ×2 | Jetson Orin NX 16GB 또는 Mac Mini M4 16GB | $1,000–1,600 |
| 10GbE 스위치 | 8-port 10GbE | $300–600 |
| **Phase 0 합계** | | **≈ $6,300–10,200** |

비교: DGX Spark 1대 ≈ $4,699(추론·LoRA용, 273GB/s 대역폭 한계) — 학습 처리량은 4090이 우위, 큰 통합메모리는 본 워크로드에 미사용.

---

## 10. 결정 요인 정리

- **학습 노드 = CUDA/Triton** (4090). 통합메모리 크기는 무관.
- **추론 = 스트림당 고정 상태** (엣지 8–16GB). 길이 늘어도 메모리 평평.
- **NAS = 분리 지점**. 머신 분리 + 하드 공유 + 온프레미스 완결.
- 큰 통합메모리 박스는 *남의 대형 VLM 의존* 시에만 정당화 → 우리 비전 모델로 대체하면 불필요.
