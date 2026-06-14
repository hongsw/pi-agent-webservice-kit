---
name: validity-gate
description: >
  AutoResearch에서 학습 전에 config의 유효성을 검증하는 Skill. growing-memory 설계공간에서
  샘플한 config가 동치/shape 규칙(논문 충실성)을 만족하는지 작은 차원으로 먼저 통과시켜,
  무효·무의미 조합이 비싼 학습에 진입하지 못하게 막는다. "config 검증", "탐색공간 가지치기",
  "스윕 전 점검"이 필요할 때 사용.
---

# validity-gate Skill

## 목적
설계 §3.4의 **유효성 게이트**를 실행한다. 샘플된 config를 학습 비용 없이 거른다:

1. **구조/shape 검사** — `d_model % n_heads == 0`, `top_k`는 `ssc` 전용, `segment_len>0`,
   `init_mode=checkpoint`는 `ssl.encoder` 필요 등.
2. **동치 테스트(실물)** — `growing-memory-pytorch`가 있으면 *작은 차원*으로 동치/shape
   테스트를 호출. 없으면 자동 skip(구조 게이트만 적용).

게이트를 통과한 config만 ASHA 학습 루프로 진입한다 → 탐색 예산 낭비 방지.

## 사용법
```bash
# 단일 config 검증 (JSON)
python3 skills/validity-gate/scripts/check.py --cfg '{"base_rule":"linear","aggregation":"ssc","segmentation":"constant","init_mode":"independent","segment_len":256,"d_model":768,"n_heads":7}'

# run config(yaml)의 탐색공간에서 N개 샘플해 통과율 측정
python3 skills/validity-gate/scripts/check.py --space tutorial/autoresearch/config/run_example.yaml -n 200
```

## 입력 / 출력
- 입력: 단일 cfg(JSON) **또는** run config(yaml)의 `space` + 샘플 수 `n`.
- 출력: 통과/탈락 여부와 사유, 공간 샘플 시 통과율·탈락 사유 집계.

## 절차 (에이전트가 따르는 단계)
1. 검증 대상 결정(단일 cfg / 공간 샘플).
2. `autoresearch.validity_gate.validity_gate(cfg)` 호출.
3. 탈락 사유를 분류해 탐색공간 축소 힌트 제시(예: `n_heads` 후보가 `d_model` 약수가 아님).

## 체크리스트
- [ ] 통과율이 너무 낮으면(<50%) 탐색공간 후보를 조정했는가?
- [ ] `ssc`가 아닌데 `top_k`를 넣은 조합을 제거했는가?
- [ ] 실물 패키지 연결 시 동치 테스트가 실제로 실행되는가(skip 아님)?

관련: [../leaderboard-analysis/SKILL.md](../leaderboard-analysis/SKILL.md) ·
[../../wiki/02-skills.md](../../wiki/02-skills.md)
