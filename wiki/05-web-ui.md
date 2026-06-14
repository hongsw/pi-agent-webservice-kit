# 05. Web UI 제공

> Web UI는 사용자가 AutoResearch 스윕을 제어하고 리더보드 결과를 확인하는
> 웹 인터페이스입니다.
> Python 표준 라이브러리만 사용하므로 추가 설치 없이 실행됩니다.
> [과제안내](../과제안내.md) · [README](../README.md)

---

## 1. 무설치 실행

```bash
# 저장소 루트에서 실행
python3 web/server.py

# 기본 포트: 8080
# 브라우저에서 http://localhost:8080 접속
```

의존성: **Python 3.8 이상** + 표준 라이브러리만 (`http.server`, `json`, `pathlib`, `subprocess`)
외부 패키지 설치 불필요. pip install 없음.

---

## 2. 화면 구성

```
┌─────────────────────────────────────────────────────────────────────┐
│  AutoResearch 대시보드                              [새로고침] [설정] │
├────────────────────────┬────────────────────────────────────────────┤
│  스윕 런처              │  리더보드                                   │
│  ┌──────────────────┐  │  ┌────────────────────────────────────┐    │
│  │ Config YAML 경로  │  │  │ rank │ run_id │ encoder │ proxy   │    │
│  │ [──────────────] │  │  │──────────────────────────────────│    │
│  │                  │  │  │  1   │ run_042│ vjepa   │ 0.873   │    │
│  │ [스윕 시작] [중단]│  │  │  2   │ run_041│ dinov2  │ 0.821   │    │
│  └──────────────────┘  │  │  3   │ run_040│ vicreg  │ 0.795   │    │
│                        │  └────────────────────────────────────┘    │
│  현재 상태              │                                            │
│  ┌──────────────────┐  │  [Best Config Export]                      │
│  │ run_id: run_042  │  │                                            │
│  │ trial: t_009     │  ├────────────────────────────────────────────┤
│  │ step: 7200/16000 │  │  run 상태 조회                              │
│  │ proxy: 0.812     │  │  Run ID: [──────────] [조회]               │
│  │ rung: 1          │  │  ┌────────────────────────────────────┐    │
│  └──────────────────┘  │  │ status / step / proxy / rung       │    │
│                        │  └────────────────────────────────────┘    │
└────────────────────────┴────────────────────────────────────────────┘
```

### 주요 UI 섹션

| 섹션 | 설명 |
|------|------|
| **스윕 런처** | config YAML 경로를 지정하고 스윕을 시작/중단 |
| **현재 상태** | 실행 중인 run/trial의 실시간 진행 상황 (3초 폴링) |
| **리더보드** | 상위 trial 목록 (proxy_score 기준 정렬, 최대 20개) |
| **Best Config Export** | 현재 best config를 YAML로 다운로드 |
| **run 상태 조회** | run_id를 입력해 특정 run의 상세 상태 조회 |

---

## 3. REST API 엔드포인트

`web/server.py`가 제공하는 REST API:

| 메서드 | 경로 | 설명 | 응답 |
|--------|------|------|------|
| `GET` | `/` | 대시보드 HTML 반환 | HTML |
| `GET` | `/api/leaderboard` | 리더보드 상위 목록 | JSON 배열 |
| `GET` | `/api/run/{run_id}` | 특정 run 상태 조회 | JSON 객체 |
| `POST` | `/api/sweep/start` | 스윕 시작 | JSON (run_id, pid) |
| `POST` | `/api/sweep/stop` | 스윕 중단 | JSON (status) |
| `GET` | `/api/export/best` | best config YAML 다운로드 | YAML 파일 |
| `GET` | `/api/status` | 서버 상태 확인 | JSON |

### API 상세: GET /api/leaderboard

```bash
# 요청
GET /api/leaderboard?n=5&metric=proxy_score

# 응답
{
  "leaderboard": [
    {
      "rank": 1,
      "run_id": "run_042",
      "trial_id": "t_007",
      "cfg": {
        "base_rule": "titans",
        "aggregation": "grm",
        "encoder": "vjepa",
        "invariance_coeff": 0.25
      },
      "proxy_score": 0.873,
      "full_score": 0.841,
      "rung": 2,
      "cost": 16000,
      "status": "done",
      "ts": "2026-06-14T11:30:00Z"
    }
  ],
  "total_trials": 47,
  "last_updated": "2026-06-14T12:00:00Z"
}
```

### API 상세: POST /api/sweep/start

```bash
# 요청
POST /api/sweep/start
Content-Type: application/json

{"config": "tutorial/autoresearch/config/run_example.yaml"}

# 응답
{
  "status": "started",
  "run_id": "run_20260614_120000",
  "pid": 12345
}
```

### API 상세: GET /api/run/{run_id}

```bash
# 요청
GET /api/run/run_20260614_120000

# 응답
{
  "run_id": "run_20260614_120000",
  "trial_id": "t_009",
  "status": "running",
  "current_step": 7200,
  "total_steps": 16000,
  "proxy_score_so_far": 0.812,
  "rung": 1,
  "elapsed_sec": 432
}
```

---

## 4. server.py 구조

```python
# web/server.py (구조 개요)

import http.server, json, pathlib, subprocess

LEADERBOARD_PATH = pathlib.Path("tutorial/autoresearch/leaderboard.jsonl")
MCP_PATH = pathlib.Path("web/mcp/autoresearch_mcp.py")

class AutoResearchHandler(http.server.BaseHTTPRequestHandler):

    def do_GET(self):
        if self.path == "/":
            self._serve_html()
        elif self.path.startswith("/api/leaderboard"):
            self._handle_leaderboard()
        elif self.path.startswith("/api/run/"):
            run_id = self.path.split("/")[-1]
            self._handle_run_status(run_id)
        elif self.path == "/api/export/best":
            self._handle_export_best()
        # ...

    def do_POST(self):
        if self.path == "/api/sweep/start":
            self._handle_sweep_start()
        elif self.path == "/api/sweep/stop":
            self._handle_sweep_stop()
        # ...

    def _call_mcp(self, tool, params):
        """MCP 서버에 JSON-RPC 요청 (subprocess + stdio)"""
        # autoresearch_mcp.py를 subprocess로 호출
        # ...

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    server = http.server.HTTPServer(("", port), AutoResearchHandler)
    print(f"서버 시작: http://localhost:{port}")
    server.serve_forever()
```

---

## 5. 프론트엔드 구조 (static/index.html)

`web/static/index.html`은 단일 HTML 파일로 구성됩니다.
외부 CDN이나 번들러 없이 동작하는 순수 HTML + CSS + JavaScript입니다.

```
web/static/index.html
├── <head>: 인라인 CSS (그리드 레이아웃, 테이블 스타일)
├── <body>:
│   ├── 헤더 (타이틀, 새로고침 버튼)
│   ├── 스윕 런처 패널 (form 태그, fetch API)
│   ├── 현재 상태 패널 (3초 자동 폴링)
│   ├── 리더보드 테이블 (동적 렌더링)
│   └── run 상태 조회 패널
└── <script>: 인라인 JavaScript
    ├── fetchLeaderboard()   → GET /api/leaderboard
    ├── startSweep()         → POST /api/sweep/start
    ├── stopSweep()          → POST /api/sweep/stop
    ├── fetchRunStatus()     → GET /api/run/{id}
    └── setInterval(fetchLeaderboard, 3000)  // 3초 폴링
```

---

## 6. 실행 예시 (단계별)

```bash
# 1. 저장소 루트에서 Web UI 서버 시작
cd /path/to/pi-agent-webservice-kit
python3 web/server.py
# → "서버 시작: http://localhost:8080" 출력

# 2. 브라우저에서 http://localhost:8080 접속

# 3. 스윕 런처에서 "스윕 시작" 버튼 클릭
#    (별도 터미널에서 loop.py가 백그라운드 실행됨)

# 4. 3초마다 리더보드 자동 갱신 확인

# 5. "Best Config Export" 버튼으로 best_config.yaml 다운로드
```

---

## 7. 포트 변경 및 기타 설정

```bash
# 포트 변경
PORT=9090 python3 web/server.py

# 리더보드 경로 변경 (환경변수)
LEADERBOARD_PATH=/tmp/my_leaderboard.jsonl python3 web/server.py
```

---

## 8. 체크리스트

- [ ] `python3 web/server.py`로 서버를 시작할 수 있다.
- [ ] 브라우저에서 리더보드를 확인할 수 있다.
- [ ] REST API 5가지 엔드포인트와 역할을 설명할 수 있다.
- [ ] Web UI가 MCP 서버(`autoresearch_mcp.py`)와 어떻게 통신하는지 설명할 수 있다.
- [ ] "Best Config Export"를 통해 YAML 파일을 다운로드할 수 있다.

---

## 관련 문서

- [00. 큰그림 · 진행순서](./00-overview.md)
- [03. MCP 연결](./03-mcp.md)
- [04. Pi Extension](./04-pi-extension.md)
- [06. 시스템 구조](./06-architecture.md)
