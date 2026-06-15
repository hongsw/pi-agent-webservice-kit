# 도커 서버 — 미연결 노드(NAS · Edge) 실구현

설계문서의 NAS(§7)와 엣지 배포(§8)를 **실제 실행되는 도커 서버**로 구현한다(이전엔 stub/placeholder).
학습 노드(4090, `run.py`)는 별도이며, 여기 두 서버가 그 **데이터 소스**와 **배포 타깃**을 제공한다.

```
[수집 시뮬]→ NAS(ar-nas)  ──HTTP manifest/shards──►  [학습 노드 run.py]
                                                           │ export best_config.json
                                                           ▼  (exports 볼륨)
                                                     Edge(ar-edge) ── 상수메모리 스트리밍 추론
```

## 기동
```bash
docker compose -f docker/docker-compose.yml up -d --build
```

## NAS 서버 (`ar-nas`, :8090) — 순수 stdlib
커밋된 샤드만 노출(설계 §7: 쓰는 중 파일 미접근). 시작 시 합성 factory 샤드 시드.
```bash
curl localhost:8090/health
curl localhost:8090/manifest                       # 커밋 샤드 jsonl
curl localhost:8090/shards/shard_0000.jsonl | head # 실제 factory 레코드
curl -X POST localhost:8090/commit                 # 새 샤드 커밋(수집 시뮬)
```
**학습 루프 연결**: config의 `data.manifest`를 `http://localhost:8090/manifest`로 지정하면
`data_interface`가 실제 커밋 샤드를 read(검증: committed_shards=8, total_samples=2048).

## Edge 서버 (`ar-edge`, :8091) — numpy 상수메모리 스트리밍
`/exports/best_config.json`(학습 export 산출물)을 로드해 토큰 스트리밍 추론. 상태 S,z만 갱신(O(1)).
```bash
curl localhost:8091/info                                  # cfg·layer별 상태 바이트(상수)
curl -X POST localhost:8091/infer -d '{"tokens":[1,2,3,4,5]}'
curl localhost:8091/memory_demo                           # L=64..8192 상태 바이트 상수 실증
```
**핸드오프**: 학습 노드의 best 번들을 exports 볼륨에 두면 엣지가 로드.
```bash
docker cp runs/export/best_config.json ar-edge:/exports/ && docker restart ar-edge
```

## 검증된 동작
- NAS: manifest/샤드 HTTP 제공, 학습 data_interface가 커밋샤드 8개(2048샘플) read.
- Edge: best_config 로드(cfg_source=export), 스트리밍 추론, **상태 135,168B가 길이 무관 상수**
  (torch 참조구현과 동일 수치). 재귀식은 torch `forward_recurrent`와 동일.

## 한계 / 다음
- Edge는 numpy 랜덤가중 참조 런타임 — 학습된 가중치(state_dict) export·로드는 후속(현재는 cfg로 구조 재현).
- NAS 샤드는 합성 factory 레코드 — 실 공장 데이터 연결 시 같은 인터페이스로 교체.
- 진짜 NFS가 필요하면 nas 서비스를 NFS export로 바꾸고 학습 노드에서 mount(인터페이스 동일).
