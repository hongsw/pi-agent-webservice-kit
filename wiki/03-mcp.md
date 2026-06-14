# 03. MCP로 외부 도구 연결

> MCP(Model Context Protocol)는 에이전트가 외부 시스템(파일, DB, API, 서비스)에
> 표준화된 방식으로 접근하도록 해주는 프로토콜입니다.
> 이 문서는 AutoResearch 시나리오의 `autoresearch-mcp` 서버를 설명합니다.
> [과제안내](../과제안내.md) · [README](../README.md)

---

## 1. MCP란?

MCP는 에이전트와 외부 도구 사이의 **표준 인터페이스**입니다.

```
Pi 에이전트
    │  MCP 프로토콜 (JSON-RPC over stdio)
    ▼
MCP 서버 (autoresearch_mcp.py)
    │
    ├── 리더보드 JSONL 파일 (로컬 파일시스템)
    ├── NAS 마운트 경로     (공유 스토리지)
    └── run 상태 딕셔너리   (메모리 / 상태 파일)
```

MCP를 쓰는 이유:
- 에이전트 코드와 도구 구현을 분리 (관심사 분리)
- 동일한 도구를 여러 에이전트에서 재사용 가능
- Pi 런타임이 도구 호출을 안전하게 샌드박스 처리

---

## 2. autoresearch-mcp 개요

### 위치
`web/mcp/autoresearch_mcp.py`

### 연결 방식 (stdio)

MCP 서버는 **stdio(표준 입출력)** 방식으로 에이전트와 통신합니다.
별도의 네트워크 포트를 열지 않고, 프로세스 파이프로 JSON-RPC 메시지를 주고받습니다.

```bash
# Pi 런타임이 내부적으로 실행하는 방식 (참고용)
python3 web/mcp/autoresearch_mcp.py
```

Pi 에이전트 설정에서 다음과 같이 등록합니다:

```yaml
mcp_servers:
  - name: autoresearch-mcp
    command: python3
    args: ["web/mcp/autoresearch_mcp.py"]
    transport: stdio
```

---

## 3. 도구(Tool) 목록

`autoresearch-mcp`가 제공하는 도구:

### 3.1 leaderboard_top

**목적**: 리더보드 상위 N개 trial 조회

| 항목 | 내용 |
|------|------|
| 입력 | `n: int` (기본값 5), `metric: str` ("proxy_score" 또는 "full_score") |
| 출력 | trial 요약 리스트 (run_id, trial_id, cfg, 점수, rung, cost, ts) |
| 데이터 소스 | 리더보드 JSONL (last-write-wins 정책) |

```json
// 호출 예시
{
  "tool": "leaderboard_top",
  "params": {"n": 3, "metric": "proxy_score"}
}

// 응답 예시
{
  "result": [
    {"run_id": "run_042", "trial_id": "t_007",
     "cfg": {"base_rule": "titans", "encoder": "vjepa"},
     "proxy_score": 0.873, "rung": 2, "cost": 16000}
  ]
}
```

---

### 3.2 leaderboard_write

**목적**: trial 결과를 리더보드 JSONL에 기록

| 항목 | 내용 |
|------|------|
| 입력 | `run_id`, `trial_id`, `cfg` (dict), `proxy_score`, `full_score` (선택), `cost`, `rung`, `status`, `code_commit` (선택), `seed` (선택) |
| 출력 | `{"written": true, "ts": "2026-06-14T12:00:00Z"}` |
| 쓰기 정책 | 동일 (run_id, trial_id) 키가 있으면 마지막 항목이 유효 (last-write-wins) |

```json
// 호출 예시
{
  "tool": "leaderboard_write",
  "params": {
    "run_id": "run_042",
    "trial_id": "t_008",
    "cfg": {"base_rule": "swla", "aggregation": "ssc",
            "encoder": "dinov2", "invariance_coeff": 0.1},
    "proxy_score": 0.761,
    "cost": 1000,
    "rung": 0,
    "status": "done"
  }
}
```

---

### 3.3 nas_list_shards

**목적**: NAS(공유 스토리지)에 커밋된 데이터 샤드 목록 조회

| 항목 | 내용 |
|------|------|
| 입력 | `manifest_path: str` (NAS 마운트 내 manifests/ 경로), `filter_status: str` (선택, "ready"/"partial") |
| 출력 | 샤드 목록 (shard_id, path, size_mb, commit_ts, status) |
| 데이터 소스 | NAS 마운트 경로의 매니페스트 파일 |

```json
// 응답 예시
{
  "result": [
    {"shard_id": "shard_001", "path": "/nas/shards/shard_001.pt",
     "size_mb": 512, "commit_ts": "2026-06-14T08:00:00Z", "status": "ready"},
    {"shard_id": "shard_002", "path": "/nas/shards/shard_002.pt",
     "size_mb": 498, "commit_ts": "2026-06-14T09:00:00Z", "status": "ready"}
  ]
}
```

---

### 3.4 run_status

**목적**: 특정 run 또는 trial의 현재 상태 조회

| 항목 | 내용 |
|------|------|
| 입력 | `run_id: str`, `trial_id: str` (선택) |
| 출력 | 상태 정보 (status, current_step, total_steps, proxy_score_so_far, rung, elapsed_sec) |
| 데이터 소스 | 메모리 내 상태 딕셔너리 또는 상태 파일 |

```json
// 응답 예시
{
  "result": {
    "run_id": "run_042",
    "trial_id": "t_009",
    "status": "running",
    "current_step": 7200,
    "total_steps": 16000,
    "proxy_score_so_far": 0.812,
    "rung": 1,
    "elapsed_sec": 432
  }
}
```

---

### 3.5 leaderboard_get

**목적**: 특정 run_id+trial_id의 전체 레코드 조회

| 항목 | 내용 |
|------|------|
| 입력 | `run_id: str`, `trial_id: str` |
| 출력 | 해당 레코드 전체 필드 또는 `{"found": false}` |

---

## 4. 리더보드 JSONL 포맷

리더보드는 **JSONL**(JSON Lines) 형식으로 저장됩니다.
각 줄이 하나의 trial 기록입니다.

```jsonl
{"run_id":"run_042","trial_id":"t_007","cfg":{"base_rule":"titans","aggregation":"grm","segmentation":"logarithmic","init_mode":"independent","segment_len":128,"top_k":16,"encoder":"vjepa","invariance_coeff":0.25,"positive_pair":"augment"},"proxy_score":0.873,"full_score":0.841,"cost":16000,"rung":2,"status":"done","code_commit":"abc1234","seed":42,"ts":"2026-06-14T11:30:00Z"}
{"run_id":"run_042","trial_id":"t_008","cfg":{"base_rule":"swla","aggregation":"ssc","encoder":"dinov2","invariance_coeff":0.1},"proxy_score":0.761,"cost":1000,"rung":0,"status":"done","ts":"2026-06-14T11:32:00Z"}
```

**last-write-wins 정책**: 동일 `(run_id, trial_id)` 키가 여러 줄 있으면
**마지막으로 기록된 줄**이 유효합니다.
이 방식으로 상태 갱신(진행 중 → 완료)을 단순하게 구현합니다.

---

## 5. MCP 서버 구현 구조 (autoresearch_mcp.py)

```python
# web/mcp/autoresearch_mcp.py (구조 개요)

import sys, json

TOOLS = {
    "leaderboard_top": handle_leaderboard_top,
    "leaderboard_write": handle_leaderboard_write,
    "nas_list_shards": handle_nas_list_shards,
    "run_status": handle_run_status,
    "leaderboard_get": handle_leaderboard_get,
}

def main():
    # stdio 루프: 한 줄씩 JSON-RPC 요청 읽기 → 도구 실행 → 응답 쓰기
    for line in sys.stdin:
        request = json.loads(line)
        tool_name = request["tool"]
        params = request.get("params", {})
        result = TOOLS[tool_name](**params)
        print(json.dumps({"result": result}), flush=True)

if __name__ == "__main__":
    main()
```

---

## 6. MCP vs Skill 비교

| 항목 | MCP 도구 | Skill |
|------|---------|-------|
| 역할 | 외부 시스템 접근 (I/O) | 절차·지식·판단 로직 |
| 상태 | stateful (파일/DB) | stateless (입력→출력) |
| 구현 언어 | Python (독립 프로세스) | SKILL.md (선언적) + 선택적 스크립트 |
| 통신 방식 | JSON-RPC over stdio | Pi 런타임 내부 호출 |
| 예시 | 리더보드 읽기/쓰기 | best config 분석, validity 검사 |

---

## 7. 체크리스트

- [ ] MCP가 에이전트와 외부 시스템을 분리하는 이유를 설명할 수 있다.
- [ ] `autoresearch-mcp`의 5가지 도구 이름과 목적을 말할 수 있다.
- [ ] JSONL 리더보드의 last-write-wins 정책이 무엇인지 설명할 수 있다.
- [ ] stdio 방식 MCP 서버를 Pi 에이전트에 등록하는 방법을 안다.
- [ ] MCP 도구와 Skill의 차이점을 2가지 이상 설명할 수 있다.

---

## 관련 문서

- [00. 큰그림 · 진행순서](./00-overview.md)
- [01. Pi Agent 기초](./01-pi-agent-basics.md)
- [02. Skill 설계](./02-skills.md)
- [05. Web UI](./05-web-ui.md)
- [06. 시스템 구조](./06-architecture.md)
- [08. 용어집](./08-glossary.md)
