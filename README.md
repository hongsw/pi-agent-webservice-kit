# AutoResearch 노드 — Pi AI Agent 웹 서비스

> **growing-memory 설계공간 × SSL 표현축을 자동으로 스윕(ASHA)해, 이 공장 데이터에
> 맞는 최적 (표현→기억) 구성을 찾아내는 단일 GPU 노드** — 를 Pi 기반 AI Agent
> 웹 서비스(Skill · MCP · Pi Extension · Web UI)로 구현한 기말 프로젝트.

karpathy의 [autoresearch](https://github.com/karpathy/autoresearch)(에이전트가 `train.py`를
고치고 `val_bpb`로 keep/revert 래칫) 원 컨셉을 **제조 도메인 + growing-memory 탐색공간**으로
일반화한다. 자세한 매핑은 [`wiki/00-overview.md`](wiki/00-overview.md) 참조.

> 📋 과제 요구사항·평가 기준 → [`과제안내.md`](과제안내.md)

---

## 🧩 키트 4요소 매핑

| 요소 | 이 프로젝트에서 | 코드 |
|---|---|---|
| **Pi Agent** | AutoResearch 오케스트레이터(스윕 운영·리더보드 질의·best 추천) | `tutorial/autoresearch/` |
| **Skill** | `validity-gate`(학습 전 동치/shape 검증), `leaderboard-analysis`(best·프록시 신뢰도) | `skills/` |
| **MCP** | `autoresearch-mcp`(리더보드 read/write/get, NAS 샤드, run 상태, export) | `web/mcp/` |
| **Pi Extension** | `autoresearch-ext`(sweep start/stop, best export) | `pi-extension/` |
| **Web UI** | 리더보드 대시보드 + 스윕 런처(무설치 stdlib 서버) | `web/` |

---

## 🚀 빠른 시작 (로컬 · 무설치)

torch / `growing-memory-pytorch` 가 없어도 **mock 백엔드**로 전 과정이 동작한다(설계 T0).

```bash
# 1) AutoResearch 스윕 실행 (게이트 → ASHA → 리더보드 → best)
cd tutorial/autoresearch
python3 run.py run --config config/run_example.yaml
python3 run.py top -n 5
python3 run.py export                       # best config 번들 export

# 2) Web UI (CLI 불가 요구사항 충족)
cd ..                                        # 저장소 루트
python3 web/server.py                        # http://localhost:8765

# 3) Skill 직접 실행
python3 skills/validity-gate/scripts/check.py --space tutorial/autoresearch/config/run_example.yaml -n 200
python3 skills/leaderboard-analysis/scripts/analyze.py -n 10

# 4) MCP 서버 점검
python3 web/mcp/autoresearch_mcp.py --selftest

# 5) Pi Extension 명령
python3 pi-extension/autoresearch_ext.py sweep-start
python3 pi-extension/autoresearch_ext.py export-best
```

### karpathy 원형 체험 (3파일 ratchet 루프)
```bash
cd tutorial/autoresearch/lab
python3 run_lab.py --iterations 30          # train.py 수정 → val_bpb → keep/revert
```
`program.md`(연구방향) · `prepare.py`(불변 평가) · `train.py`(에이전트가 고치는 유일 파일) 구조.

---

## 🖥️ 4090 노드 배포 (실학습)

실제 학습은 RTX 4090 머신에서 수행한다(설계 §2). 배포·실행은 한 줄:

```bash
scripts/deploy_4090.sh run      # rsync 후 4090에서 스윕 실행
scripts/deploy_4090.sh web      # 4090에서 대시보드 기동
```
`AR_HOST`(기본 `martin@linux-builder`)·`AR_DEST`로 대상 변경. GPU 가용성은 실행 로그의
`[autoresearch] GPU: {...cuda_op_ok: True}` 로 확인된다.

**growing-memory 실물 연결:** 머신에서 `export GROWING_MEMORY_HOME=<repo경로>` 설정 시
`model_adapter`가 실물 `build`/동치테스트를 호출하고, 없으면 mock으로 폴백한다.

---

## 📂 구조

```
pi-agent-webservice-kit/
├── tutorial/autoresearch/        ← AutoResearch 코어(Pi Agent 본체)
│   ├── autoresearch/             · search_space · validity_gate · controller_asha
│   │                               proxy · leaderboard · ratchet · model_adapter
│   │                               data_interface · export · loop · config_io
│   ├── config/run_example.yaml   · §9 실행 설정(탐색공간·rung·프록시)
│   ├── run.py                    · CLI(run/top/export)
│   └── lab/                      · karpathy 3파일 ratchet 원형(program/prepare/train)
├── web/  server.py · static/     ← Web UI 대시보드 + REST
│   └── mcp/autoresearch_mcp.py   ← MCP 서버(stdio)
├── skills/                       ← validity-gate · leaderboard-analysis
├── pi-extension/                 ← autoresearch-ext (manifest + 명령)
├── scripts/deploy_4090.sh        ← 4090 배포/실행
├── wiki/ (00~08)                 ← 개념·아키텍처 위키
└── report/                       ← 기말 보고서
```

---

## 🔧 기술 스택
- **언어/런타임:** Python 3.10+ (코어는 순수 stdlib · 무설치 실행), 실학습 시 PyTorch 2.x + CUDA
- **탐색:** ASHA(Asynchronous Successive Halving) · 유효성 게이트 · 프록시(factory_mqar / short_horizon)
- **인터페이스:** MCP(JSON-RPC stdio) · http.server 대시보드 · JSONL 리더보드(last-write-wins)
- **모델(설계):** `growing-memory-pytorch`(base_rule/aggregation/segmentation) × SSL(V-JEPA/DINOv2/VICReg)

## 🐳 NAS · Edge 도커 서버 (실연결)
설계의 NAS(§7)·엣지 배포(§8)를 **실행되는 도커 서버**로 구현. `docker compose -f docker/docker-compose.yml up -d --build`
→ NAS(:8090)가 커밋 샤드를 HTTP로 제공(학습 루프가 실제 read), Edge(:8091)가 best_config 로드 후
상수메모리 스트리밍 추론(상태 135KB 길이 무관). → [`docker/README.md`](docker/README.md)

## 🧠 메모리 캐싱 최적화 (RNN 추론)
선형 어텐션 계열(linear/dla/titans)은 **학습은 병렬(O(L²)), 추론은 고정 상태 RNN 재귀(O(1) 상태)**로
동치 변환된다 → 엣지에서 길이 무관 평평한 메모리. 4090 실측: 병렬 vs 재귀 출력 차이 ~1e-7,
L=4096에서 메모리 ~20×↓, L=16384 병렬 OOM에도 재귀는 135KB 상수 상태로 동작.
검증(병렬 에이전트): [`tutorial/autoresearch/RECURRENT.md`](tutorial/autoresearch/RECURRENT.md) ·
`python3 tutorial/autoresearch/verify_recurrent.py --rule dla --stress --bench`

## 🔬 grokking 튜닝 (8변형 회상 학습)
8변형(base_rule×aggregation) 모두 MQAR 연관회상을 학습. 정확회상 grok 난이도는 메커니즘에
좌우됨 — softmax/영속메모리/게이팅/(감쇠+용량)은 grok(→1.0), 순수 선형은 다중키 천장(~0.5).
1키 회상은 전부 1.0. → [`tutorial/autoresearch/GROKKING.md`](tutorial/autoresearch/GROKKING.md) ·
`python3 tutorial/autoresearch/grok_tune.py --variants all --pairs 2`

## 📚 더 읽기
[`wiki/00-overview.md`](wiki/00-overview.md) · [`wiki/06-architecture.md`](wiki/06-architecture.md) ·
[`report/final-project-template.md`](report/final-project-template.md) · Pi Docs <https://pi.dev/docs/latest>
