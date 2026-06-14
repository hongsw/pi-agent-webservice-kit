# 06. 시스템 구조

> 이 문서는 AutoResearch 노드의 전체 시스템 구조를 설계 관점에서 설명합니다.
> karpathy autoresearch 원 컨셉에서 출발해 이 키트의 아키텍처로 이어지는
> 설계 결정과 컴포넌트 관계를 다룹니다.
> [과제안내](../과제안내.md) · [README](../README.md)

---

## §1. 시스템 내 위치

이 AutoResearch 노드는 제조 AI 파이프라인의 **학습/탐색 단계**에 위치합니다.

```
[Edge 노드]          [NAS]              [AutoResearch 노드]       [배포 노드]
  카메라·센서  →→→→  shards/   →→→→  Pi Agent + 스윕 루프  →→→→  추론 서버
                    manifests/         리더보드 대시보드
                    checkpoints/       Web UI (port 8080)
```

**입력**: NAS에 커밋된 데이터 샤드 + run config YAML
**출력**: 최적 (표현→기억) config YAML + 리더보드 (JSONL)

---

## §2. karpathy 원형 → 이 키트 설계 결정

### 원형 ratchet 루프

```
[karpathy autoresearch]

program.md (연구 방향)
    ↓
에이전트가 train.py 편집
    ↓
prepare.py로 평가 (val_bpb)
    ↓
개선? → git commit (keep)
나빠짐? → git reset HEAD~1 (revert)
    ↓
반복 (~12 exp/hour, 밤새 ~100회)
```

### 이 키트의 일반화

원 컨셉의 한계(creativity ceiling: 즉시 개선만 수용 → 탐색적 후퇴 불가)를
**ASHA + 프록시 평가** 로 극복합니다.

```
[이 키트 AutoResearch 루프]

run config YAML (karpathy의 program.md 역할)
    ↓
ASHA 컨트롤러: 다음 trial config 생성
    ↓
validity-gate Skill: 무효 config 조기 차단
    ↓
model_adapter: 학습 실행 (growing-memory-pytorch or mock)
    ↓
proxy 평가 (factory_mqar / short_horizon_pred)
    ↓
leaderboard_write MCP: JSONL 기록 (last-write-wins)
    ↓
ratchet.py: best 갱신 시 code_commit 해시 기록 (karpathy git commit 역할)
    ↓
ASHA 승급 판정 (rung: 1k→4k→16k 스텝)
    ↓
leaderboard-analysis Skill: best 선정, 이상 감지
    ↓
반복
```

### lab/ — 원형 직접 체험

```
tutorial/autoresearch/lab/
├── program.md    연구 방향 (사람이 작성, karpathy 방식)
├── prepare.py    불변 평가 (val_bpb 형 지표, mock 데이터)
├── train.py      가변 단일 파일 (에이전트/학생이 수정)
└── run_lab.py    ratchet 러너 (git keep/revert 시뮬레이션)
```

`lab/`은 원 컨셉을 순수하게 보여주는 최소 재현입니다.
`ratchet.py`는 `lab/run_lab.py`의 로직을 프로덕션 레이어로 구현합니다.

---

## §3. 컴포넌트 다이어그램

```
┌──────────────────────────────────────────────────────────────────────┐
│                    AutoResearch 노드 컴포넌트                         │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                    Pi Agent 레이어                           │    │
│  │                                                             │    │
│  │  loop.py ←──────────── Pi 런타임 (시스템 프롬프트)           │    │
│  │     │                                                       │    │
│  │     ├── Skill: validity-gate   (config 사전 검증)           │    │
│  │     ├── Skill: leaderboard-analysis (best 선정)             │    │
│  │     ├── MCP: autoresearch-mcp (stdio, JSON-RPC)             │    │
│  │     └── Extension: autoresearch-ext (sweep start/stop)      │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                │                                                     │
│                ▼                                                     │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                  AutoResearch 코어 레이어                    │    │
│  │                                                             │    │
│  │  controller_asha.py   ASHA 컨트롤러 (trial 생성·승급)       │    │
│  │  ratchet.py           keep-or-revert (best 추적)            │    │
│  │  search_space.py      탐색 공간 정의                        │    │
│  │  model_adapter.py     growing-memory-pytorch / mock         │    │
│  │  proxy.py             factory_mqar / short_horizon_pred     │    │
│  │  leaderboard.py       JSONL 읽기/쓰기 (last-write-wins)     │    │
│  │  data_interface.py    NAS 샤드 접근                         │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                │                                                     │
│                ▼                                                     │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                     Web 레이어                              │    │
│  │                                                             │    │
│  │  web/server.py        stdlib http.server, REST API          │    │
│  │  web/static/index.html 리더보드 대시보드 (순수 HTML/JS)      │    │
│  │  web/mcp/autoresearch_mcp.py  MCP 서버 (stdio)              │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                │                                                     │
│                ▼                                                     │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                    스토리지 레이어                           │    │
│  │                                                             │    │
│  │  leaderboard.jsonl    trial 결과 기록                       │    │
│  │  NAS shards/          공장 데이터 샤드                      │    │
│  │  NAS manifests/       샤드 매니페스트                       │    │
│  │  NAS checkpoints/     모델 체크포인트                       │    │
│  └─────────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────┘
```

---

## §4. AutoResearch 루프 상세 (ASHA 컨트롤러)

### ASHA 동작 원리

ASHA(Async Successive Halving Algorithm)는 다수의 저예산 trial에서 시작해
성능이 좋은 것만 더 높은 예산(rung)으로 승급시키는 조기중단 알고리즘입니다.

```
Rung 0 (1,000 스텝): trial_01, trial_02, trial_03, trial_04, trial_05, trial_06
                            ↓ 상위 1/3 승급 ↓
Rung 1 (4,000 스텝):        trial_02,              trial_04,              trial_06
                                    ↓ 상위 1/3 승급 ↓
Rung 2 (16,000 스텝):               trial_02,              trial_06
                                           ↓ 최종 best
                                           trial_02 (best config)
```

### config 탐색 공간 (`search_space.py`)

**시퀀스 축 (기억 구조)**:

| 파라미터 | 선택지 |
|---------|--------|
| `base_rule` | `linear`, `swla`, `dla`, `titans` |
| `aggregation` | `residual`, `grm`, `soup`, `ssc` |
| `segmentation` | `constant`, `logarithmic` |
| `init_mode` | `checkpoint`, `independent` |
| `segment_len` | 32, 64, 128, 256 |
| `top_k` | 8, 16, 32 |

**표현 축 (SSL 인코더)**:

| 파라미터 | 선택지 |
|---------|--------|
| `encoder` | `vjepa`, `dinov2`, `vicreg` |
| `invariance_coeff` | 0.0, 0.1, 0.25, 0.5 |
| `positive_pair` | `augment`, `temporal` |

전체 그리드 크기: 4×4×2×2×4×3×3×4×2 = 약 18,432가지
→ ASHA로 저예산에서 다수 샘플링 후 유망한 것만 승급

---

## §5. 프록시 평가

프록시 평가는 full 학습(16k 스텝) 없이 저예산(1k 스텝)에서 성능을 추정합니다.
상관이 높은 프록시일수록 early stopping 효율이 증가합니다.

### factory_mqar (주 프록시)

**MQAR**(Multi-Query Associative Recall) — 합성 recall 과제.
입력 시퀀스에 (key, value) 쌍을 제시한 후, 나중에 key를 보여주고 value를 맞추는 과제.
시퀀스 메모리 구조의 recall 능력을 직접 측정합니다.

```
입력: [(k1,v1), (k2,v2), ..., (k_n,v_n), ?, k_3]
출력: v_3  (recall)
```

점수: 정확도 (0~1), 높을수록 우수

### short_horizon_pred (보조 프록시)

자기지도(self-supervised) 예측 오차. 짧은 시퀀스에서 다음 토큰 예측 손실.
인코더의 표현 품질과 메모리 구조의 통합 능력을 간접 측정합니다.

점수: 예측 오차 (낮을수록 우수) → 내부에서 1-error로 정규화해 proxy_score로 통일

### 프록시↔full 상관 모니터링

`leaderboard-analysis` Skill이 Spearman ρ를 계산합니다:
- ρ ≥ 0.7 → proxy_trust = "high" (ASHA 신뢰 가능)
- 0.5 ≤ ρ < 0.7 → proxy_trust = "medium" (주의)
- ρ < 0.5 → proxy_trust = "low" → 경고 발행 (프록시 재검토 필요)

---

## §6. 메모리 예산 관리 및 병렬 스윕

### 메모리 예산

| rung | 스텝 수 | 예상 GPU 메모리 | 예상 소요 시간 (A100 기준) |
|------|---------|---------------|--------------------------|
| 0 | 1,000 | ~8 GB | ~3분 |
| 1 | 4,000 | ~8 GB | ~12분 |
| 2 | 16,000 | ~8 GB | ~45분 |

### 병렬 스윕 전략

단일 GPU 노드에서 순차 실행이 기본입니다.
다중 GPU 환경에서는 `controller_asha.py`가 동시 실행 trial 수를 조절합니다.

```python
# config/run_example.yaml 내 병렬도 설정
max_concurrent_trials: 1   # 단일 GPU: 1
rungs: [1000, 4000, 16000]
eta: 3                     # 각 rung에서 상위 1/eta 승급
```

---

## §7. NAS 인터페이스 (`data_interface.py`)

NAS 데이터 접근 레이어. 실물 NAS 마운트 없이 mock으로도 동작합니다.

```python
class DataInterface:
    def list_shards(self, manifest_path) -> List[ShardInfo]:
        """커밋된 샤드 목록 반환 (NAS 매니페스트 파싱)"""

    def sample_batch(self, shard_ids=None) -> Batch:
        """학습/검증 배치 샘플링
           NAS 없음 → 합성 데이터로 폴백"""

    def get_shard_stats(self) -> Dict:
        """샤드 통계 (총 샘플 수, 시간 범위 등)"""
```

mock 모드: `shard_ids=None`이면 합성 공장 데이터를 생성해 반환.
실물 모드: NAS 마운트 경로에서 `.pt` 파일 로드.

---

## §8. 산출물 및 핸드오프

스윕 완료 후 산출물:

| 산출물 | 경로 | 내용 |
|--------|------|------|
| 리더보드 | `leaderboard.jsonl` | 전체 trial 기록 (JSONL) |
| best config | `best_config.yaml` | 최적 (표현→기억) 구성 YAML |
| 체크포인트 | `NAS/checkpoints/best/` | best trial 모델 가중치 |

**핸드오프**: `best_config.yaml`과 체크포인트를 배포 노드로 전달 →
추론 서버가 동일 config로 `growing-memory-pytorch` 모델을 로드해 서빙.

---

## §9. model_adapter 폴백 전략

```python
# tutorial/autoresearch/autoresearch/model_adapter.py

try:
    import growing_memory_pytorch as gm
    # 실물 growing-memory-pytorch 사용
    adapter = RealModelAdapter(gm)
except ImportError:
    # mock 폴백: 순수 stdlib, GPU 불필요
    # ToyModel로 루프 폐쇄(loop closure) 검증
    adapter = MockModelAdapter()
```

mock ToyModel은 랜덤 proxy_score를 반환하지만,
**루프 전체 흐름**(validity-gate → 학습 → leaderboard 기록 → ASHA 승급)의
정합성을 검증하는 데 사용합니다.

---

## 체크리스트

- [ ] karpathy ratchet과 이 키트 ASHA 루프의 대응 관계를 설명할 수 있다.
- [ ] ASHA의 rung 구조(3단계: 1k→4k→16k)를 도식으로 설명할 수 있다.
- [ ] 탐색 공간(시퀀스 축 6개 파라미터 × 표현 축 3개 파라미터)을 나열할 수 있다.
- [ ] factory_mqar와 short_horizon_pred가 각각 무엇을 측정하는지 설명할 수 있다.
- [ ] proxy_trust가 "low"일 때 에이전트가 어떤 행동을 해야 하는지 설명할 수 있다.
- [ ] model_adapter 폴백 전략의 목적을 설명할 수 있다.
- [ ] 스윕 완료 후 배포 노드로 전달하는 산출물 2가지를 말할 수 있다.

---

## 관련 문서

- [00. 큰그림 · 진행순서](./00-overview.md)
- [01. Pi Agent 기초](./01-pi-agent-basics.md)
- [02. Skill 설계](./02-skills.md)
- [03. MCP 연결](./03-mcp.md)
- [07. 참고문헌](./07-resources.md)
- [08. 용어집](./08-glossary.md)
