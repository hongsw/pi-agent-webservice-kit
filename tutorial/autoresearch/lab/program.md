# program.md — 연구 방향 (사람이 작성·반복)

> karpathy/autoresearch의 `program.md`에 대응. 에이전트(여기선 Pi AutoResearch 에이전트)가
> 매 실험 전에 읽는 **연구 방향 지시서**다. 사람이 편집하고, 에이전트는 이 방향을 따른다.
> 이 키트에서는 상위 레이어의 `config/run_example.yaml`(ASHA 탐색공간)이 이 역할을 일반화한다.

## 목표
`train.py` 한 파일만 수정하여 **val_bpb(validation bits-per-byte)** 를 낮춘다.
val_bpb는 vocab 크기에 독립이라 아키텍처 변경을 공정하게 비교할 수 있다(낮을수록 우수).

## 규칙 (karpathy ratchet)
- 매 실험: `train.py` 수정 → 고정 예산으로 학습 → val_bpb 평가.
- val_bpb가 좋아지면 **keep**(스냅샷/커밋), 나빠지면 **revert**(직전 스냅샷 복원).
- `prepare.py`는 **수정 금지**(데이터 준비 + 평가의 불변 기준).

## 탐색 우선순위 (이 공장 데이터 기준 가설)
1. 모델 폭(width)·깊이(depth)의 균형 — 과/소용량 모두 손해.
2. 정규화(dropout) — 저데이터에서 과적합 억제.
3. 학습률(lr) — 안정 수렴 구간 탐색.

## 피해야 할 것
- 한 번에 여러 knob 동시 변경(원인 분리 불가).
- prepare.py의 평가 기준 변경(부정 비교).

## 한계 (원본 명시)
ratchet은 "즉시 개선"만 수용 → 탐색적 후퇴 불가("creativity ceiling").
이 한계를 넘는 일반화가 상위 레이어의 ASHA 다중탐색 + 프록시 조기중단이다.
