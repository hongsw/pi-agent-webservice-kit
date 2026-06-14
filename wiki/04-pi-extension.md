# 04. Pi Extension 확장

> Pi Extension은 Pi 플랫폼에 새로운 **명령(command)** 을 추가하여
> 에이전트가 플랫폼 수준의 동작(프로세스 제어, 파일 export 등)을 수행하도록 확장합니다.
> 이 문서는 `autoresearch-ext`의 구조와 제공 명령을 설명합니다.
> [과제안내](../과제안내.md) · [README](../README.md)

---

## 1. Pi Extension이란?

Pi Extension은 MCP 도구(데이터 I/O)나 Skill(절차 지식)과 달리,
**Pi 플랫폼 자체의 기능을 확장**하는 메커니즘입니다.

```
Pi 에이전트
├── Skill      → "무엇을 어떻게 판단하라"는 절차 지식
├── MCP 도구   → 외부 시스템(파일/DB/API)에 읽기/쓰기
└── Extension  → 플랫폼 수준 명령 (프로세스 제어, 이벤트 발행, export)
```

Extension을 통해 에이전트는 단순한 텍스트 생성을 넘어,
**실제 시스템을 제어**하는 행동을 할 수 있습니다.

---

## 2. autoresearch-ext 개요

### 위치
`pi-extension/` 디렉터리

### 역할

| 명령 | 설명 |
|------|------|
| `sweep start` | AutoResearch 스윕 루프 시작 (loop.py 프로세스 기동) |
| `sweep stop` | 실행 중인 스윕 루프 중단 (graceful shutdown) |
| `export best` | 현재 리더보드의 best config를 YAML 파일로 export |

---

## 3. Extension 구조

```
pi-extension/
├── manifest.json      Extension 메타데이터 및 명령 선언
└── commands/
    ├── sweep_start.py  sweep start 명령 구현
    ├── sweep_stop.py   sweep stop 명령 구현
    └── export_best.py  export best 명령 구현
```

### manifest.json 예시

```json
{
  "name": "autoresearch-ext",
  "version": "0.1.0",
  "description": "AutoResearch 스윕 제어 및 best config export 명령 확장",
  "commands": [
    {
      "name": "sweep",
      "subcommands": [
        {
          "name": "start",
          "description": "AutoResearch ASHA 스윕 루프를 시작합니다.",
          "params": [
            {"name": "config", "type": "string", "description": "run config YAML 경로", "required": false,
             "default": "tutorial/autoresearch/config/run_example.yaml"}
          ],
          "handler": "commands/sweep_start.py"
        },
        {
          "name": "stop",
          "description": "실행 중인 스윕 루프를 중단합니다.",
          "params": [
            {"name": "run_id", "type": "string", "description": "중단할 run ID (없으면 현재 실행 중인 run)", "required": false}
          ],
          "handler": "commands/sweep_stop.py"
        }
      ]
    },
    {
      "name": "export",
      "subcommands": [
        {
          "name": "best",
          "description": "현재 리더보드에서 best config를 YAML로 export합니다.",
          "params": [
            {"name": "metric", "type": "string", "description": "기준 지표 (proxy_score 또는 full_score)", "required": false, "default": "proxy_score"},
            {"name": "output", "type": "string", "description": "출력 파일 경로", "required": false, "default": "best_config.yaml"}
          ],
          "handler": "commands/export_best.py"
        }
      ]
    }
  ]
}
```

---

## 4. 각 명령 상세

### 4.1 sweep start

**목적**: `tutorial/autoresearch/run.py`를 백그라운드 프로세스로 기동하여
AutoResearch ASHA 스윕 루프를 시작합니다.

**동작 흐름**:

```
1. config YAML 경로 확인 (없으면 기본값 사용)
2. 이미 실행 중인 스윕이 있는지 PID 파일 확인
   → 있으면 "이미 실행 중" 오류 반환
3. python3 tutorial/autoresearch/run.py --config {config} 를 subprocess로 기동
4. PID 파일 (.autoresearch_pid) 기록
5. run_id 생성 (타임스탬프 기반) → 반환
```

**반환 예시**:
```json
{
  "status": "started",
  "run_id": "run_20260614_120000",
  "pid": 12345,
  "config": "tutorial/autoresearch/config/run_example.yaml"
}
```

---

### 4.2 sweep stop

**목적**: 실행 중인 스윕 루프를 안전하게(graceful) 중단합니다.

**동작 흐름**:

```
1. PID 파일에서 프로세스 PID 읽기
2. SIGTERM 시그널 전송 (graceful shutdown)
   → loop.py 내부: SIGTERM 수신 시 현재 trial 완료 후 종료
3. 5초 대기 → 프로세스 종료 확인
4. 미종료 시 SIGKILL (강제 종료)
5. PID 파일 삭제
6. 최종 리더보드 상태 반환
```

**반환 예시**:
```json
{
  "status": "stopped",
  "run_id": "run_20260614_120000",
  "trials_completed": 23,
  "best_proxy_score": 0.873
}
```

---

### 4.3 export best

**목적**: 현재 리더보드에서 best trial의 config를 YAML 파일로 export합니다.
이 파일을 배포 노드에 가져가 추론 서버를 기동하는 데 사용합니다.

**동작 흐름**:

```
1. autoresearch-mcp leaderboard_top 호출 (n=1, metric={metric})
2. best trial의 cfg 딕셔너리 추출
3. 메타데이터 추가:
   - export_ts (export 시각)
   - source_run_id, source_trial_id
   - source_proxy_score, source_full_score
   - code_commit (학습 시 코드 커밋 해시)
4. YAML 직렬화 → {output} 경로에 저장
5. 파일 경로와 내용 요약 반환
```

**export된 YAML 예시**:
```yaml
# best_config.yaml — AutoResearch export
# export_ts: 2026-06-14T12:00:00Z
# source: run_20260614_120000 / t_007
# proxy_score: 0.873, full_score: 0.841
# code_commit: abc1234

base_rule: titans
aggregation: grm
segmentation: logarithmic
init_mode: independent
segment_len: 128
top_k: 16
encoder: vjepa
invariance_coeff: 0.25
positive_pair: augment
```

---

## 5. Extension을 Web UI에서 호출하는 방법

Web UI(`web/server.py`)는 Extension 명령을 REST API로 노출합니다.

| REST 엔드포인트 | Extension 명령 |
|----------------|---------------|
| `POST /api/sweep/start` | `autoresearch-ext sweep start` |
| `POST /api/sweep/stop` | `autoresearch-ext sweep stop` |
| `GET /api/export/best` | `autoresearch-ext export best` |

```javascript
// Web UI에서 스윕 시작 예시 (fetch API)
const response = await fetch('/api/sweep/start', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({config: 'tutorial/autoresearch/config/run_example.yaml'})
});
const data = await response.json();
console.log('run_id:', data.run_id);
```

---

## 6. Extension vs MCP vs Skill 비교

| 항목 | Extension | MCP 도구 | Skill |
|------|-----------|---------|-------|
| 주요 역할 | 플랫폼 명령 (프로세스 제어) | 외부 시스템 I/O | 절차·판단 지식 |
| 트리거 | 에이전트 명령 실행 | 에이전트 도구 호출 | 에이전트 절차 실행 |
| 상태 변경 | O (프로세스 기동/종료) | O (파일 읽기/쓰기) | X (순수 로직) |
| 예시 | sweep start/stop | leaderboard_write | validity 검사 |

---

## 7. 체크리스트

- [ ] Pi Extension이 MCP 도구·Skill과 다른 점을 설명할 수 있다.
- [ ] `autoresearch-ext`가 제공하는 3가지 명령(sweep start/stop, export best)을 설명할 수 있다.
- [ ] `manifest.json`의 역할과 필수 필드를 안다.
- [ ] `export best`가 반환하는 YAML에 포함되는 메타데이터 항목을 말할 수 있다.
- [ ] Web UI에서 Extension 명령을 REST API로 호출하는 방법을 안다.

---

## 관련 문서

- [00. 큰그림 · 진행순서](./00-overview.md)
- [01. Pi Agent 기초](./01-pi-agent-basics.md)
- [03. MCP 연결](./03-mcp.md)
- [05. Web UI](./05-web-ui.md)
- [06. 시스템 구조](./06-architecture.md)
