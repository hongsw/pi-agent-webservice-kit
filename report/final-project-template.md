# 기말 프로젝트 보고서 — AutoResearch 노드: Pi AI Agent로 운영하는 제조 AI 자율 탐색 시스템

> Pi AI Agent 웹 서비스  
> 작성자: ___ / 학번: ___ / 날짜: ___  
> 이 보고서는 [과제안내](../과제안내.md) 제출물 §2 기준에 따라 작성합니다.  
> `___` 로 표시된 부분은 학생이 직접 채워야 합니다.

---

## 1. 선택한 시나리오

**Training / AutoResearch 노드**

온프레미스 제조 AI 시스템에서, 단일 GPU 노드가 `growing-memory-pytorch`의
설계공간(시퀀스 메모리 축 × SSL 표현 축)을 ASHA(Successive Halving) 알고리즘으로
자동 스윕하여 "이 공장 데이터에 맞는 최적 (표현→기억) 구성"을 찾아내는 AutoResearch 컨트롤러를
**Pi AI Agent 웹 서비스로 노출**하는 프로젝트입니다.

### 원천 개념: karpathy/autoresearch

이 시나리오는 Andrej Karpathy의 **autoresearch** 아이디어를 제조 AI에 맞게 일반화합니다.

| karpathy autoresearch | 이 프로젝트 (제조 + growing-memory) |
|---|---|
| train.py 자유 편집 | 검증된 config 공간 샘플링 (validity-gate로 논문 충실성 보장) |
| 단일 에이전트 순차 ratchet | ASHA 다중 config 병렬 + 조기중단 |
| 고정 5분 예산 | rung (1k→4k→16k 스텝) |
| val_bpb (validation bits-per-byte) | 프록시 (factory_mqar recall / short_horizon_pred) → 상위만 full 평가 |
| keep (git commit) / revert (git reset HEAD~1) | 리더보드 last-write-wins + best 추적 (code_commit 해시 기록) |
| program.md (연구 방향) | run config YAML + Pi 에이전트 지시 |
| prepare.py (불변 평가) | NAS 데이터 인터페이스 + 불변 프록시/full 평가 |

karpathy 원형의 한계("creativity ceiling" — 즉시 개선만 수용해 탐색적 후퇴 불가)를
ASHA 다중 탐색으로 극복합니다.

---

## 2. 문제 정의

### 문제 상황

공장에서 수집된 센서/비디오 데이터를 분석하는 AI 모델을 개발할 때,
어떤 표현 인코더(V-JEPA / DINOv2 / VICReg)와
어떤 시퀀스 메모리 구조(Titans / SWLA / DLA / Linear × 집약 방식 4종)를
조합해야 이 공장 데이터에 가장 적합한지 **사람이 일일이 실험하기 어렵습니다**.

탐색 공간만 해도 4×4×2×2×4×3×3×4×2 ≈ 18,000가지 이상이며,
각 실험에 수십 분이 소요됩니다.

### 해결 접근

1. **ASHA 자동 탐색**: 저예산(1k 스텝)에서 다수를 시도하고, 유망한 것만 고예산(16k 스텝)으로 승급
2. **validity-gate**: 학습 전 무효 config를 차단해 GPU 낭비 방지
3. **프록시 평가**: factory_mqar(합성 recall)로 1k 스텝에서 최종 성능을 사전 추정
4. **Pi AI Agent**: 전체 루프를 자율 운영하고, 사용자의 자연어 질의에 응답
5. **Web UI**: ML 엔지니어가 스윕을 제어하고 리더보드를 실시간 모니터링

### 핵심 가설

> "factory_mqar proxy_score가 높은 (표현→기억) config는
>  full 학습(16k 스텝) 후에도 공장 데이터에서 우수한 성능을 보인다."

검증 방법: Rung 2 승급 trial에서 proxy↔full Spearman ρ 계산 (목표: ρ ≥ 0.7)

---

## 3. 서비스 대상 사용자

| 사용자 유형 | 역할 | 주요 니즈 |
|------------|------|----------|
| **제조 ML 엔지니어** | 스윕 실행·모니터링 | 실시간 리더보드, best config 확인 |
| **AI 연구자** | 탐색 공간 설계, 프록시 신뢰도 검토 | proxy↔full 상관 분석 |
| **시스템 관리자** | 스윕 시작/중단, 배포 핸드오프 | best config YAML export |

**주 사용자**: 공장 데이터를 다루는 ML 엔지니어. AI 논문 전문 지식 없이도
Web UI를 통해 스윕을 실행하고 결과를 확인할 수 있어야 합니다.

---

## 4. 핵심 기능

| 번호 | 기능 | 설명 |
|------|------|------|
| F-1 | ASHA 자동 스윕 | config 탐색 공간에서 trial을 생성·승급·종료 |
| F-2 | validity-gate 사전 필터 | 무효 config를 학습 전 차단 (GPU 낭비 방지) |
| F-3 | proxy 평가 | factory_mqar로 1k 스텝 내 성능 추정 |
| F-4 | 리더보드 관리 | JSONL 기록, last-write-wins, best 추적 |
| F-5 | ratchet 추적 | best 갱신 시 code_commit 해시 기록 |
| F-6 | 자연어 Q&A | Pi 에이전트가 리더보드 질의에 자연어로 응답 |
| F-7 | Web UI 대시보드 | 리더보드 테이블, 실시간 run 상태, 스윕 런처 |
| F-8 | best config export | 최적 config를 YAML로 다운로드 (배포 핸드오프) |
| F-9 | 이상 감지 | proxy↔full 상관 낮으면 경고 발행 |

---

## 5. 시스템 구조

```
┌──────────────────────────────────────────────────────────────────────┐
│                    AutoResearch 노드                                  │
│                                                                      │
│  ┌───────────────────────────────────────────────┐                  │
│  │              Pi Agent 레이어                  │                  │
│  │  loop.py → Pi 런타임 → Skill/MCP/Extension   │                  │
│  └───────────────────────────────────────────────┘                  │
│                │                                                     │
│  ┌─────────────▼─────────────────────────────────┐                  │
│  │           AutoResearch 코어 레이어             │                  │
│  │  controller_asha  ratchet  search_space       │                  │
│  │  model_adapter    proxy    leaderboard        │                  │
│  │  data_interface                               │                  │
│  │  lab/ (karpathy 원형 재현)                    │                  │
│  └───────────────────────────────────────────────┘                  │
│                │                                                     │
│  ┌─────────────▼─────────────────────────────────┐                  │
│  │              Web 레이어                       │                  │
│  │  web/server.py  →  web/static/index.html     │                  │
│  │  web/mcp/autoresearch_mcp.py (MCP, stdio)    │                  │
│  └───────────────────────────────────────────────┘                  │
│                │                                                     │
│  ┌─────────────▼─────────────────────────────────┐                  │
│  │           스토리지 레이어                      │                  │
│  │  leaderboard.jsonl  NAS shards/  checkpoints/ │                  │
│  └───────────────────────────────────────────────┘                  │
└──────────────────────────────────────────────────────────────────────┘
```

### 주요 파일 구조

```
tutorial/autoresearch/
  run.py                     진입점
  config/run_example.yaml    스윕 설정
  autoresearch/
    search_space.py          탐색 공간 정의
    validity_gate.py         validity-gate 구현
    model_adapter.py         growing-memory / mock 폴백
    proxy.py                 factory_mqar, short_horizon_pred
    leaderboard.py           JSONL 읽기/쓰기
    data_interface.py        NAS 샤드 접근
    controller_asha.py       ASHA 알고리즘
    ratchet.py               keep-or-revert (karpathy 원형)
    loop.py                  Pi 에이전트 오케스트레이션 루프
  lab/
    program.md               연구 방향 (karpathy 방식)
    prepare.py               불변 평가 (val_bpb 형 지표)
    train.py                 가변 단일 파일
    run_lab.py               ratchet 러너

web/
  server.py                  stdlib http.server, REST API
  static/index.html          리더보드 대시보드
  mcp/
    autoresearch_mcp.py      MCP 서버 (stdio)

skills/
  validity-gate/SKILL.md     Skill 명세
  leaderboard-analysis/SKILL.md

pi-extension/
  manifest.json              Extension 메타데이터
  commands/                  sweep start/stop, export best
```

---

## 6. Skill / MCP / Pi Extension 활용 방식

### 6.1 Skill: validity-gate

**역할**: 학습 시작 전 config의 동치·shape 검증으로 무효 config 조기 차단

**활용 위치**: `loop.py` 루프의 2단계 — trial config 생성 직후, 학습 실행 전

**수행 절차**:
1. config 필수 키 존재 확인
2. 값 범위/열거 검증 (base_rule ∈ {linear,swla,dla,titans} 등)
3. 동치 검사 — 리더보드에 동등한 config가 이미 있는지 확인
4. Tensor shape 사전 검증 — `model_adapter.dry_run()`으로 shape 불일치 탐지

**효과**: GPU 낭비 없이 탐색 효율 향상. 이 키트에서 무효 config 차단율: ___% (학생이 측정)

---

### 6.2 Skill: leaderboard-analysis

**역할**: 리더보드 분석 → best config 선정 + proxy↔full 순위 상관 점검

**활용 위치**: `loop.py` 루프의 7단계 — 각 trial 완료 후

**수행 절차**:
1. JSONL 읽기 (last-write-wins 처리)
2. 완료 trial 필터, 점수 정렬
3. Spearman ρ 계산 (proxy↔full)
4. proxy_trust 판정 ("high"/"medium"/"low")
5. 이상 감지 및 경고 생성

---

### 6.3 MCP: autoresearch-mcp

**위치**: `web/mcp/autoresearch_mcp.py` (stdio JSON-RPC)

**제공 도구**:

| 도구 | 호출 위치 | 목적 |
|------|----------|------|
| `leaderboard_top` | Web UI, Pi Agent | 상위 trial 조회 |
| `leaderboard_write` | loop.py (각 trial 완료) | trial 결과 기록 |
| `nas_list_shards` | loop.py (초기화) | 커밋된 데이터 샤드 목록 |
| `run_status` | Web UI | 실시간 run 상태 조회 |
| `leaderboard_get` | Pi Agent Q&A | 특정 trial 상세 조회 |

---

### 6.4 Pi Extension: autoresearch-ext

**위치**: `pi-extension/`

**제공 명령**:

| 명령 | 트리거 | 동작 |
|------|--------|------|
| `sweep start` | Web UI "스윕 시작" / 자연어 "스윕 시작해줘" | `run.py` 백그라운드 기동 |
| `sweep stop` | Web UI "중단" / 자연어 "스윕 멈춰줘" | SIGTERM → graceful shutdown |
| `export best` | Web UI "Best 다운로드" | best config YAML 생성 |

---

## 7. Web UI 설명

### 실행 방법

```bash
python3 web/server.py
# http://localhost:8080 접속 (추가 설치 없음)
```

### 화면 구성

| 섹션 | 기능 |
|------|------|
| **스윕 런처** | config YAML 경로 입력 → 스윕 시작/중단 버튼 |
| **현재 상태** | 실행 중인 run/trial의 실시간 진행 상황 (3초 폴링) |
| **리더보드 테이블** | 상위 trial 목록 (proxy_score 기준, 3초 자동 갱신) |
| **Best Config Export** | YAML 다운로드 → 배포 노드에 전달 |
| **run 상태 조회** | run_id 입력 → 상세 상태 조회 |

### REST API 요약

| 메서드 | 경로 | 설명 |
|--------|------|------|
| GET | `/` | 대시보드 HTML |
| GET | `/api/leaderboard` | 리더보드 상위 목록 |
| GET | `/api/run/{run_id}` | run 상태 조회 |
| POST | `/api/sweep/start` | 스윕 시작 |
| POST | `/api/sweep/stop` | 스윕 중단 |
| GET | `/api/export/best` | best config YAML |

---

## 8. 구현 결과

### 8.1 프로토타입 구현 내역

| 컴포넌트 | 구현 상태 | 비고 |
|---------|----------|------|
| ASHA 컨트롤러 | ✅ 구현 | `controller_asha.py` |
| validity-gate Skill | ✅ 구현 | `skills/validity-gate/SKILL.md` |
| leaderboard-analysis Skill | ✅ 구현 | `skills/leaderboard-analysis/SKILL.md` |
| autoresearch-mcp | ✅ 구현 | `web/mcp/autoresearch_mcp.py` |
| autoresearch-ext | ✅ 구현 | `pi-extension/` |
| Web UI 대시보드 | ✅ 구현 | `web/server.py`, `web/static/index.html` |
| lab/ (karpathy 원형) | ✅ 구현 | `tutorial/autoresearch/lab/` |
| ratchet.py | ✅ 구현 | best 추적, code_commit 기록 |
| model_adapter (실물) | ⚠️ 선택 | growing-memory-pytorch 설치 필요 |
| model_adapter (mock) | ✅ 구현 | ToyModel, 순수 stdlib |

### 8.2 루프 폐쇄 검증 (mock 모드)

mock ToyModel을 사용한 루프 폐쇄 검증 결과:

- trial 생성 → validity-gate → 학습 → leaderboard 기록 → ASHA 승급 전체 흐름 정상 동작
- 스윕 30회 trial에서 Rung 0→1→2 승급 정상 확인
- Web UI에서 리더보드 실시간 갱신 확인

```
[실행 결과 스크린샷 또는 로그 일부를 여기에 삽입]
___
```

### 8.3 주요 수치 (학생이 실제 실행 후 채울 것)

| 지표 | 값 |
|------|-----|
| 총 trial 수 (mock 30회 기준) | ___ |
| validity-gate 차단 비율 | ___% |
| Rung 0→1 승급 비율 | ___% (이론: 33%) |
| Rung 1→2 승급 비율 | ___% (이론: 33%) |
| proxy↔full Spearman ρ (mock) | ___ (mock은 랜덤) |
| Web UI 응답 시간 | ___ ms |

---

## 9. 한계점 및 개선 방향

### 9.1 현재 한계

| 한계 | 설명 |
|------|------|
| **실물 growing-memory 미통합** | mock ToyModel만 검증. 실제 GPU 환경에서 `growing-memory-pytorch` 설치 후 검증 필요 |
| **프록시↔full 상관 보정 미구현** | proxy_trust "low" 시 경고만 발행, 자동 프록시 재선택은 미구현 |
| **NAS 실물 미연결** | mock 데이터로 테스트. 실물 NAS 마운트 경로 설정 필요 |
| **단일 GPU 순차 실행** | 다중 GPU 병렬화 미구현 (controller_asha.py의 max_concurrent_trials=1) |
| **Pi 에이전트 실물 연결** | loop.py는 에이전트 루프를 시뮬레이션. 실제 Pi 런타임 연결 필요 |

### 9.2 개선 방향

**T1 단계 (단기 — 기본 통합)**:
- [ ] `growing-memory-pytorch` 실물 설치 및 `model_adapter.py` 실물 경로 연결
- [ ] 실물 NAS 마운트 경로 설정 (`data_interface.py`)
- [ ] Pi 런타임에 에이전트 등록, 실제 자연어 Q&A 테스트

**T2 단계 (중기 — 성능 개선)**:
- [ ] 프록시↔full 상관 자동 보정: Spearman ρ 낮으면 프록시 함수 자동 교체
- [ ] 다중 GPU 병렬 스윕 (`max_concurrent_trials` 증가)
- [ ] 탐색 공간 축소: validity-gate에서 통계 기반 bad region 자동 배제

**T3 단계 (장기 — 고급 기능)**:
- [ ] karpathy ratchet + ASHA 혼합 전략: 유망 영역에서는 ratchet, 미탐색 영역에서는 ASHA
- [ ] 다중 공장 데이터셋 동시 탐색 (멀티태스킹)
- [ ] Web UI에 탐색 공간 시각화 (t-SNE / UMAP 리더보드 플롯)

### 9.3 배운 점 (학생이 직접 작성)

```
이 프로젝트를 통해 배운 것:
___

Pi AI Agent를 활용하면서 느낀 점:
___

karpathy autoresearch 원 컨셉과 ASHA 일반화의 차이에서 배운 것:
___
```

---

## 참고 문헌

- karpathy/autoresearch: <https://github.com/karpathy/autoresearch>
- Pi 공식 문서: <https://pi.dev/docs/latest>
- Agent Skills: <https://agentskills.io/specification>
- ASHA: Li et al. (2018), arXiv:1810.05934
- V-JEPA: Assran et al. (2024), arXiv:2404.08471
- DINOv2: Oquab et al. (2023), arXiv:2304.07193
- VICReg: Bardes et al. (2022), arXiv:2105.04906
- Balestriero & LeCun, SSL closed-form: arXiv:2205.11508
- MQAR: Zoology, arXiv:2302.06555
- Titans: Ali et al. (2025), arXiv:2501.00663

---

*이 보고서 템플릿은 [과제안내](../과제안내.md)의 §2 제출물 기준에 따라 작성되었습니다.*
*`___` 표시 부분은 실제 실험 결과 및 본인 경험으로 반드시 채워주세요.*
