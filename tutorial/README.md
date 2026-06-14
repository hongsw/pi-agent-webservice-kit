# 실습 (tutorial)

AutoResearch 노드의 **동작하는 최소 구현**. torch / `growing-memory-pytorch` 없이도
mock 백엔드로 끝까지 돈다(설계 T0: 루프 폐쇄 검증).

## autoresearch/ — Pi Agent 본체(AutoResearch 오케스트레이터)
```bash
cd autoresearch
python3 run.py run --config config/run_example.yaml   # 스윕: 게이트→ASHA→리더보드→best
python3 run.py top -n 5                                # 리더보드 상위
python3 run.py export                                  # best config 번들 export
```
| 모듈 | 역할(설계 §) |
|---|---|
| `search_space.py` | §3 탐색공간(시퀀스×표현축) + 이론 가지치기 |
| `validity_gate.py` | §3.4 학습 전 동치/shape 검증 |
| `controller_asha.py` | §4 ASHA 컨트롤러(저예산→승급) |
| `proxy.py` | §5 프록시(factory_mqar / short_horizon) + 순위상관 |
| `leaderboard.py` | §7 JSONL 리더보드(last-write-wins) |
| `ratchet.py` | karpathy keep-or-revert 원형 |
| `model_adapter.py` | build(cfg) — 실물 growing-memory import + mock 폴백, GPU 프로브 |
| `export.py` | §8 best export(동치성 재확인) |

## lab/ — karpathy autoresearch 3파일 원형
```bash
cd autoresearch/lab
python3 run_lab.py --iterations 30        # train.py 편집 → val_bpb → keep/revert(ratchet)
```
`program.md`(연구방향·사람) · `prepare.py`(불변 평가) · `train.py`(에이전트가 고치는 유일 파일).
상위 레이어(ASHA·프록시·리더보드)가 이 원형을 다중탐색으로 일반화한다.

→ 개념은 [`../wiki/06-architecture.md`](../wiki/06-architecture.md).
