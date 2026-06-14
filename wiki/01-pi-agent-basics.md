# 01. Pi로 AI Agent 만들기

> 이 문서는 Pi를 이용해 AI Agent를 만드는 기본 개념을 설명하고,
> AutoResearch 오케스트레이터를 Pi 에이전트로 바라보는 관점을 정리합니다.
> [과제안내](../과제안내.md) · [README](../README.md)

---

## 1. Pi 에이전트란?

**Pi**는 AI Agent를 선언적으로 정의하고 실행하는 플랫폼입니다.
에이전트는 다음 4가지 요소로 구성됩니다.

```
Pi 에이전트
├── 시스템 프롬프트  (역할·제약 정의)
├── Skill 목록      (절차화된 작업 묶음)
├── MCP 도구 목록   (외부 시스템 접근)
├── Extension 명령  (플랫폼 확장 기능)
└── 대화 루프       (입력 → 추론 → 도구 호출 → 응답)
```

에이전트는 사용자 입력(자연어 또는 이벤트)을 받아 → 어떤 도구/Skill을 쓸지 추론하고 →
결과를 반환하는 **자율 루프**를 수행합니다.

---

## 2. AutoResearch 오케스트레이터를 Pi 에이전트로 보는 관점

이 키트에서 Pi 에이전트는 **AutoResearch 오케스트레이터** 역할을 합니다.
"오케스트레이터"란 여러 도구와 Skill을 조율해 복잡한 워크플로를 자율적으로 운영하는 에이전트를 말합니다.

### 2.1 에이전트의 역할

| 역할 | 설명 |
|------|------|
| **스윕 운영** | ASHA 컨트롤러를 통해 config trial을 생성·승급·종료 |
| **리더보드 Q&A** | 사용자 질문에 대해 현재 best config를 자연어로 답변 |
| **best 추천** | `leaderboard-analysis` Skill을 호출해 최적 구성 도출 |
| **이상 감지** | proxy↔full 순위 불일치 시 경고 발행 |

### 2.2 에이전트의 입출력

```
입력:
  - 자연어 질의    ("지금 best encoder는 뭐야?")
  - 이벤트         (스윕 완료, trial 실패)
  - Web UI REST   (POST /api/sweep/start)

출력:
  - 자연어 응답    ("현재 best: encoder=vjepa, proxy=0.87")
  - 도구 실행 결과 (리더보드 JSON, run 상태)
  - 파일           (best config YAML export)
```

### 2.3 에이전트 루프 (loop.py)

`tutorial/autoresearch/autoresearch/loop.py`에 구현된 루프:

```
┌──────────────────────────────────────────────────┐
│               AutoResearch 루프                  │
│                                                  │
│  1. 다음 trial config 생성 (ASHA 컨트롤러)       │
│         ↓                                        │
│  2. validity-gate Skill 호출                     │
│       → 유효: 다음 단계                          │
│       → 무효: 이 config 스킵 (ASHA에 실패 기록)  │
│         ↓                                        │
│  3. model_adapter 로 학습 실행                   │
│       (growing-memory-pytorch 또는 mock ToyModel) │
│         ↓                                        │
│  4. proxy 점수 계산 (factory_mqar, short_horizon) │
│         ↓                                        │
│  5. leaderboard_write (MCP) 호출                 │
│         ↓                                        │
│  6. ASHA 승급 판정 → rung 올리거나 trial 종료    │
│         ↓                                        │
│  7. leaderboard-analysis Skill 호출              │
│       → best config 갱신, proxy↔full 이상 감지   │
│         ↓                                        │
│  8. 다음 trial로 → 1 반복                        │
└──────────────────────────────────────────────────┘
```

---

## 3. Pi 에이전트 정의 예시

Pi의 에이전트 설정은 YAML로 선언합니다.
아래는 AutoResearch 오케스트레이터의 최소 정의 예시입니다.

```yaml
# pi-agent.yaml (참고용 예시 — Pi 플랫폼 문법에 맞게 조정 필요)
name: autoresearch-orchestrator
description: |
  제조 공장 데이터에 최적화된 growing-memory 표현/기억 구성을
  ASHA 스윕으로 자동 탐색하는 AI 에이전트.

system_prompt: |
  당신은 AutoResearch 오케스트레이터입니다.
  - 탐색 공간(시퀀스 축 × 표현 축)의 trial을 ASHA로 스케줄링합니다.
  - 학습 전 validity-gate로 무효 config를 차단합니다.
  - 리더보드에서 best config를 조회하고 추천합니다.
  - 항상 근거(trial_id, proxy_score)를 함께 제시합니다.

skills:
  - validity-gate
  - leaderboard-analysis

mcp_servers:
  - autoresearch-mcp

extensions:
  - autoresearch-ext
```

---

## 4. 에이전트 실행 내부 흐름

Pi 에이전트가 실행될 때 내부적으로 일어나는 일:

```
사용자 입력 ("현재 best config 알려줘")
    │
    ▼
[Pi 런타임] 시스템 프롬프트 + 입력 → LLM 추론
    │
    ├── "leaderboard_top 도구 호출 필요"
    │       → MCP autoresearch-mcp 호출
    │       → JSONL 리더보드에서 상위 5개 반환
    │
    ├── "leaderboard-analysis Skill 실행 필요"
    │       → Skill 절차 실행 → best config 산출
    │
    └── 최종 응답 조립 → "best: encoder=vjepa, base_rule=titans, proxy=0.87"
```

---

## 5. Pi 에이전트 vs 일반 LLM API 호출 비교

| 항목 | 일반 LLM API | Pi 에이전트 |
|------|-------------|------------|
| 상태 관리 | 없음 (stateless) | 세션/컨텍스트 지원 |
| 도구 연결 | 직접 구현 필요 | MCP 표준 프로토콜 |
| 절차 재사용 | 없음 | Skill로 패키징 |
| 플랫폼 확장 | 없음 | Extension 명령 |
| 오케스트레이션 | 수동 코딩 | 자동 루프 |
| 복잡한 워크플로 | 별도 파이프라인 필요 | 에이전트 루프 내장 |

---

## 6. 이 키트에서 에이전트와 코드의 관계

```
Pi 에이전트 (선언)
       │
       ├── tutorial/autoresearch/autoresearch/loop.py   (오케스트레이션 루프)
       ├── tutorial/autoresearch/autoresearch/controller_asha.py (ASHA 알고리즘)
       ├── tutorial/autoresearch/autoresearch/model_adapter.py   (모델 실행)
       └── tutorial/autoresearch/run.py                 (진입점)
```

실습 시 `python3 tutorial/autoresearch/run.py`를 실행하면
loop.py가 에이전트 루프를 시뮬레이션합니다.

---

## 7. 체크리스트

- [ ] Pi 에이전트의 4가지 구성 요소를 설명할 수 있다.
- [ ] AutoResearch 루프의 8단계 흐름을 순서대로 나열할 수 있다.
- [ ] Pi 에이전트가 일반 LLM API 호출과 다른 점을 3가지 이상 설명할 수 있다.
- [ ] `loop.py`가 하는 일을 한 문장으로 요약할 수 있다.
- [ ] `validity-gate`가 루프 2단계에서 실패하면 어떻게 되는지 설명할 수 있다.

---

## 관련 문서

- [00. 큰그림 · 진행순서](./00-overview.md)
- [02. Skill 설계](./02-skills.md)
- [03. MCP 연결](./03-mcp.md)
- [04. Pi Extension](./04-pi-extension.md)
- [06. 시스템 구조](./06-architecture.md)
