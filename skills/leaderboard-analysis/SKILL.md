---
name: leaderboard-analysis
description: >
  AutoResearch 리더보드를 분석해 best config를 선정하고, 프록시↔full 평가의 순위상관
  (Spearman ρ)을 점검해 프록시를 신뢰할지 판정하는 Skill. "best 뽑아줘", "리더보드 분석",
  "프록시가 믿을 만한가", "어떤 config가 좋은가"가 필요할 때 사용한다.
---

# leaderboard-analysis Skill

## 목적
설계 §5/§8을 실행한다:

1. **best 선정** — full_score 우선, 없으면 proxy_score로 상위 trial 정렬.
2. **프록시 신뢰도(proxy_trust)** — 최상위 rung에 도달한 trial들의 proxy↔full
   **Spearman 순위상관 ρ**을 계산. ρ가 낮으면 프록시가 best와 어긋난 것 → 보정 필요.
3. **추천** — best config와 그 근거(점수·비용·백엔드·커밋해시)를 요약.

## 사용법
```bash
python3 skills/leaderboard-analysis/scripts/analyze.py \
    --leaderboard tutorial/autoresearch/runs/leaderboard.jsonl -n 10
```

## 입력 / 출력
- 입력: leaderboard.jsonl 경로, 상위 표시 개수 n.
- 출력(JSON):
  - `summary` (trial 수, 총 학습비용, best)
  - `top` (상위 n trial)
  - `proxy_full_rank_corr` + `proxy_trust`(high/medium/low)
  - `recommendation` (best cfg와 근거)

## proxy_trust 판정 기준
| ρ (Spearman) | 판정 | 의미 |
|---|---|---|
| ≥ 0.7 | high | 프록시로 조기중단해도 best 보존 가능 |
| 0.4–0.7 | medium | 프록시 유지하되 상위권은 full로 재확인 |
| < 0.4 | low | 프록시 재설계(과제/평가 길이) 필요 |

## 절차
1. 리더보드 로드 → 최신 레코드(last-write-wins)만 사용.
2. 최상위 rung 도달 trial의 (proxy, full)로 ρ 계산.
3. proxy_trust 판정 + best 추천 + 회귀 방지 안내(export 전 동치성 재확인).

## 체크리스트
- [ ] best가 동치성 게이트를 통과하는가(export 안전)?
- [ ] proxy_trust가 low면 프록시 과제를 재정의했는가?
- [ ] 비용(cost) 대비 점수 효율을 함께 봤는가?

관련: [../validity-gate/SKILL.md](../validity-gate/SKILL.md) ·
[../../wiki/02-skills.md](../../wiki/02-skills.md)
