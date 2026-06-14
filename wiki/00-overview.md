# 00. 큰그림 · 진행순서

> 이 위키는 **AutoResearch 노드** 시나리오를 기준으로 Pi AI Agent 웹 서비스 키트 전체를 안내합니다.
> [과제안내](../과제안내.md) · [README](../README.md)

---

## 1. 원천 개념: karpathy/autoresearch

이 키트는 Andrej Karpathy의 **autoresearch** 아이디어를 제조 AI에 맞게 일반화한 프로젝트입니다.

### karpathy 원 컨셉

```
┌─────────────────────────────────────────────────────────────────┐
│                 karpathy/autoresearch 원형                       │
│                                                                 │
│  AI 에이전트가 train.py 단 한 파일을 자유롭게 편집               │
│  ↓ 고정 wall-clock 예산(5분) 으로 1회 실험                       │
│  ↓ 평가: val_bpb (validation bits-per-byte, 낮을수록 우수)       │
│       vocab 크기 독립 → 아키텍처 변경도 공정 비교 가능            │
│                                                                 │
│  ratchet 루프:                                                  │
│    개선되면 → keep (git commit)                                  │
│    나빠지면 → revert (git reset HEAD~1)                         │
│                                                                 │
│  밤새 ~12 experiments/hour, ~100회, 15~20개 개선 유지            │
│                                                                 │
│  3파일 구조:                                                    │
│    program.md  — 사람이 쓰는 연구 방향 (불변)                    │
│    prepare.py  — 불변: 데이터 준비 + 평가 함수                   │
│    train.py    — 에이전트가 고치는 유일한 파일 (가변)             │
│                                                                 │
│  한계: "creativity ceiling"                                     │
│    즉시 개선만 수용 → 탐색적 후퇴 불가                           │
└─────────────────────────────────────────────────────────────────┘
```

### 원 컨셉 → 이 키트 매핑

| karpathy autoresearch | 이 키트 (제조 + growing-memory) |
|---|---|
| train.py 자유 편집 | 검증된 config 공간 샘플링 (validity-gate로 논문 충실성 보장) |
| 단일 에이전트 순차 ratchet | ASHA 다중 config 병렬 + 조기중단 |
| 고정 5분 예산 | rung (1k→4k→16k 스텝) |
| val_bpb | 프록시 (factory_mqar recall / short_horizon_pred) → 상위만 full 평가 |
| keep (git commit) / revert (git reset HEAD~1) | 리더보드 last-write-wins + best 추적 (code_commit 해시 기록) |
| program.md (연구 방향) | run config (YAML) + Pi 에이전트 지시 |
| prepare.py (불변 평가) | NAS 데이터 인터페이스 + 불변 프록시/full 평가 |

### karpathy 3파일 구조의 무설치 재현: `tutorial/autoresearch/lab/`

이 키트는 원 컨셉을 그대로 체험할 수 있는 **lab** 디렉터리를 포함합니다.

```
tutorial/autoresearch/lab/
├── program.md      karpathy 방식의 연구 방향 지시 파일 (사람이 작성)
├── prepare.py      불변 평가 함수 (val_bpb 형 지표, mock 데이터)
├── train.py        에이전트(또는 학생)가 수정하는 가변 단일 파일
└── run_lab.py      ratchet 러너 (git keep/revert 시뮬레이션)
```

`lab/`은 원 컨셉을 순수하게 보여주는 최소 재현입니다.
그 **위에** `controller_asha`, `proxy`, `leaderboard`가 "일반화 레이어"로 올라갑니다.

```
tutorial/autoresearch/autoresearch/ratchet.py
    ↑ lab/run_lab.py의 keep-or-revert 로직을 프로덕션 레이어로 구현
    ↑ best 갱신 시에만 채택, git 커밋 해시 기록
```

---

## 2. 이 키트가 다루는 시나리오

**AutoResearch 노드** — 온프레미스 제조 AI 시스템에서 단일 GPU 노드가
`growing-memory-pytorch`의 설계 공간(시퀀스 메모리 축 × SSL 표현 축)을
ASHA(Successive Halving) 알고리즘으로 자동 스윕하여, 공장 데이터에 최적인
"표현 → 기억" 구성을 찾아내는 AutoResearch 컨트롤러를
**Pi AI Agent 웹 서비스로 노출**하는 프로젝트입니다.

karpathy의 "밤새 자율 실험하는 에이전트"가 곧 이 키트의 **Pi AutoResearch 에이전트**입니다.
`program.md` / run config를 읽고 ratchet/ASHA 루프를 자율 운영하는 주체로 동작합니다.

---

## 3. 제조 AI 시스템 내 위치

```
┌─────────────────────────────────────────────────────────────────┐
│                   제조 현장 (온프레미스)                          │
│                                                                 │
│  카메라/센서                                                     │
│      │                                                          │
│      ▼                                                          │
│  ┌─────────┐    NAS (공유 스토리지)                             │
│  │  Edge   │ ──────────────────────────────────────┐           │
│  │  노드   │    shards/, manifests/, checkpoints/  │           │
│  └─────────┘                                       │           │
│                                                    │           │
│  ┌─────────────────────────────────────────────────▼────────┐  │
│  │          Training / AutoResearch 노드 (★ 이 키트)        │  │
│  │                                                          │  │
│  │   Pi Agent (karpathy 방식의 자율 실험 에이전트)           │  │
│  │       │                                                  │  │
│  │       ├── lab/             원 컨셉 3파일 재현             │  │
│  │       ├── ratchet.py       keep-or-revert 로직           │  │
│  │       ├── controller_asha  ASHA 일반화 레이어             │  │
│  │       ├── Skill: validity-gate / leaderboard-analysis    │  │
│  │       ├── MCP:   autoresearch-mcp                        │  │
│  │       ├── Pi Extension: autoresearch-ext                 │  │
│  │       └── Web UI: 리더보드 대시보드 + 스윕 런처           │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌─────────┐                                                    │
│  │  배포   │  ← best config export → 추론 서버 배포             │
│  │  노드   │                                                    │
│  └─────────┘                                                    │
└─────────────────────────────────────────────────────────────────┘
```

**데이터 흐름 요약**:

| 단계 | 출발 | 도착 | 내용 |
|------|------|------|------|
| 1 | Edge 노드 | NAS | 원시 공장 데이터 샤드 커밋 |
| 2 | AutoResearch 노드 | NAS | 커밋된 샤드 목록 조회 (MCP) |
| 3 | AutoResearch 노드 | GPU | 각 trial 학습 실행 |
| 4 | GPU | 리더보드 (JSONL) | proxy/full 점수 기록 (MCP) |
| 5 | Pi Agent | Web UI | 리더보드 시각화, best 추천 |
| 6 | Web UI 사용자 | 배포 노드 | best config export → 추론 배포 |

---

## 4. 키트 4요소 매핑

| 키트 요소 | 이 시나리오에서의 역할 | 구현 파일 |
|-----------|----------------------|-----------|
| **Pi Agent** | AutoResearch 오케스트레이터 (스윕 운영, 리더보드 Q&A, best 추천) | `tutorial/autoresearch/autoresearch/loop.py` |
| **Skill** | `validity-gate` (무효 config 차단) + `leaderboard-analysis` (best 선정, 순위 상관 점검) | `skills/validity-gate/SKILL.md`, `skills/leaderboard-analysis/SKILL.md` |
| **MCP** | `autoresearch-mcp`: 리더보드 조회/기록, NAS 샤드 조회, run 상태 조회 | `web/mcp/autoresearch_mcp.py` |
| **Pi Extension** | `autoresearch-ext`: 스윕 시작/중단, best config export 명령 | `pi-extension/` |
| **Web UI** | 리더보드 대시보드 + 스윕 런처 (stdlib, 무설치) | `web/server.py`, `web/static/index.html` |

---

## 5. 키트 4요소 결합 흐름

```
사용자 (브라우저)
    │  HTTP GET/POST
    ▼
web/server.py (stdlib http.server)
    │
    ├─ GET /api/leaderboard  ──► autoresearch-mcp (leaderboard_top)
    ├─ POST /api/sweep/start ──► Pi Extension (autoresearch-ext sweep start)
    ├─ POST /api/sweep/stop  ──► Pi Extension (autoresearch-ext sweep stop)
    └─ GET /api/run/{id}     ──► autoresearch-mcp (run_status)

Pi Agent (loop.py) — karpathy ratchet의 ASHA 일반화
    ├─ ratchet.py: keep-or-revert (best 갱신 시만 채택)
    ├─ validity-gate Skill → 무효 config 조기 차단
    ├─ 학습 실행 → proxy 점수 계산
    ├─ autoresearch-mcp leaderboard_write 호출
    └─ leaderboard-analysis Skill → best 선정
```

---

## 6. 학습 순서 (권장)

```
0단계: 원 컨셉 체험        tutorial/autoresearch/lab/
        run_lab.py로 karpathy ratchet 흐름 직접 확인
        ↓
1단계: Pi 기초             wiki/01-pi-agent-basics.md
        ↓
2단계: Skill 설계          wiki/02-skills.md
        ↓
3단계: MCP 연결            wiki/03-mcp.md
        ↓
4단계: Pi Extension        wiki/04-pi-extension.md
        ↓
5단계: Web UI              wiki/05-web-ui.md
        ↓
6단계: 시스템 구조 이해     wiki/06-architecture.md
        ↓
7단계: 전체 통합 실행
        python3 tutorial/autoresearch/run.py
        python3 web/server.py
```

### 단계별 체크리스트

- [ ] karpathy ratchet의 3파일 구조(program.md/prepare.py/train.py)를 설명할 수 있다.
- [ ] val_bpb가 무엇이고 왜 vocab 독립 지표인지 설명할 수 있다.
- [ ] 이 키트가 원 컨셉을 어떻게 "일반화"하는지 매핑표로 설명할 수 있다.
- [ ] Pi 에이전트 개념과 역할을 설명할 수 있다.
- [ ] `validity-gate` Skill이 왜 필요한지 설명할 수 있다.
- [ ] `autoresearch-mcp`의 도구 목록과 입출력을 안다.
- [ ] Pi Extension이 MCP/Skill과 어떻게 다른지 설명할 수 있다.
- [ ] Web UI를 무설치로 실행하고 리더보드를 확인할 수 있다.
- [ ] ASHA 컨트롤러의 승급(rung) 구조를 도식으로 설명할 수 있다.

---

## 관련 문서

- [01. Pi Agent 기초](./01-pi-agent-basics.md)
- [02. Skill 설계](./02-skills.md)
- [03. MCP 연결](./03-mcp.md)
- [04. Pi Extension](./04-pi-extension.md)
- [05. Web UI](./05-web-ui.md)
- [06. 시스템 구조](./06-architecture.md)
- [07. 참고문헌](./07-resources.md)
- [08. 용어집](./08-glossary.md)
