# Web UI

AutoResearch 노드의 사용자 화면 — **리더보드 대시보드 + 스윕 런처**. 순수 stdlib
(`http.server`)라 무설치로 실행된다(CLI 불가 요구사항 충족).

## 실행
```bash
python3 web/server.py            # http://localhost:8765
PORT=9000 python3 web/server.py
```

## 화면
- 요약 카드(백엔드·trial 수·best full/proxy·총 학습비용·NAS 샤드)
- 리더보드 테이블(상위 trial, cfg 요약, rung/status)
- 스윕 시작 버튼 / best export 버튼 / 실행 상태 패널(라이브 폴링)

## REST API
| 메서드 | 경로 | 설명 |
|---|---|---|
| GET | `/api/summary` | 리더보드 요약 + NAS 캐시 + 백엔드 |
| GET | `/api/leaderboard?n=20` | 상위 trial |
| GET | `/api/runs` | 진행/완료 스윕 상태 |
| POST | `/api/sweep` | 스윕 시작(백그라운드) |
| POST | `/api/export` | best config export |

## mcp/ — MCP 서버
```bash
python3 web/mcp/autoresearch_mcp.py --selftest    # 도구 점검
```
도구: `leaderboard_top` · `leaderboard_summary` · `leaderboard_get` · `leaderboard_write` ·
`nas_list_shards` · `run_status` · `export_best`. 클라이언트 등록:
`{"command":"python3","args":["web/mcp/autoresearch_mcp.py"]}`

→ 상세는 [`../wiki/05-web-ui.md`](../wiki/05-web-ui.md) · [`../wiki/03-mcp.md`](../wiki/03-mcp.md).
