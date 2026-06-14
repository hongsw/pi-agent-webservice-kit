# 02. Skill 설계·활용

> Skill은 에이전트가 특정 작업을 더 정확하고 반복 가능하게 수행하도록 돕는
> 지식·절차·스크립트·리소스 묶음입니다.
> 이 문서는 AutoResearch 시나리오의 두 Skill을 상세히 설명합니다.
> [과제안내](../과제안내.md) · [README](../README.md)

---

## 1. Skill이란?

Skill은 에이전트에게 "이 상황에서는 이렇게 행동하라"는 절차를 제공하는 패키지입니다.
시스템 프롬프트가 역할을 정의한다면, Skill은 **구체적인 실행 방법**을 정의합니다.

```
Skill 패키지
├── SKILL.md     (Skill 명세: 목적·입력·절차·산출)
├── (선택) scripts/   (보조 스크립트)
└── (선택) resources/ (참조 데이터, 예시)
```

Skill은 `skills/<skill-name>/SKILL.md` 경로에 위치하며,
에이전트가 Pi 런타임에서 로드하여 사용합니다.

---

## 2. SKILL.md 구조

```markdown
# Skill: <skill-name>

## 목적
이 Skill이 해결하는 문제를 1~2문장으로 기술.

## 트리거 조건
언제 이 Skill을 호출해야 하는지 명시.

## 입력
- 파라미터 이름: 타입, 설명

## 절차
1. 첫 번째 단계
2. 두 번째 단계
...

## 산출 (출력)
- 출력 항목: 타입, 설명

## 예시
입력 예시와 예상 출력 예시.

## 주의사항
엣지 케이스, 제약 조건.
```

---

## 3. validity-gate Skill

### 위치
`skills/validity-gate/SKILL.md`

### 목적
학습을 시작하기 **전에** config의 동치·shape 검증을 수행하여,
실행 시간을 낭비하지 않고 무효 config를 조기에 차단합니다.

ASHA 스윕에서 무효 config가 학습 도중 실패하면 GPU 시간과 리더보드 슬롯이 낭비됩니다.
validity-gate는 이를 미리 방지합니다.

### 입력

| 파라미터 | 타입 | 설명 |
|---------|------|------|
| `config` | dict | 검증할 trial config (YAML에서 파싱된 딕셔너리) |
| `data_interface` | DataInterface | shape 검증에 사용할 배치 제공자 |

### 절차

```
1. config 필드 존재 확인
   - base_rule, aggregation, segmentation, encoder 등 필수 키 체크
   - 누락 시 → ValidationError("missing key: {key}")

2. 값 범위/열거 검증
   - base_rule ∈ {linear, swla, dla, titans}
   - aggregation ∈ {residual, grm, soup, ssc}
   - segmentation ∈ {constant, logarithmic}
   - encoder ∈ {vjepa, dinov2, vicreg}
   - segment_len > 0, top_k > 0, invariance_coeff ∈ [0, 1]
   - 위반 시 → ValidationError("invalid value: {field}={value}")

3. 동치(equivalence) 검사
   - (base_rule=linear, aggregation=grm)처럼 의미상 동일한 다른 config가
     이미 리더보드에 있는지 확인
   - 있으면 → DuplicateError("equivalent to run_id={id}")

4. Tensor shape 사전 검증
   - data_interface.sample_batch() 호출 → 더미 배치 획득
   - model_adapter.dry_run(config, batch) 실행 (1 스텝, GPU 없이)
   - shape 불일치 시 → ShapeError("expected {a}, got {b}")

5. 모든 검사 통과 → {"valid": true, "warnings": [...]} 반환
```

### 산출

```json
{
  "valid": true,
  "warnings": ["top_k=32 은 segment_len=64 에서 비효율적일 수 있음"],
  "checks_passed": ["fields", "ranges", "equivalence", "shape"]
}
```

실패 시:
```json
{
  "valid": false,
  "error_type": "ShapeError",
  "message": "encoder vjepa expects (B, T, 384), got (B, T, 512)"
}
```

### 예시

```python
# loop.py 내 사용 예 (개념 코드)
result = validity_gate.run(config=trial_config, data_interface=di)
if not result["valid"]:
    asha.mark_failed(trial_id, reason=result["message"])
    continue  # 이 trial 스킵, 다음 config로
```

---

## 4. leaderboard-analysis Skill

### 위치
`skills/leaderboard-analysis/SKILL.md`

### 목적
리더보드(JSONL)에서 현재까지의 trial 결과를 분석하여:
1. **best config** 선정 (proxy_score 기준 또는 full_score 기준)
2. **proxy↔full 순위 상관** 점검 (Spearman 상관계수 계산)
3. 이상 감지 — 상관이 낮으면 프록시 신뢰도 경고 발행

### 입력

| 파라미터 | 타입 | 설명 |
|---------|------|------|
| `leaderboard_path` | str | JSONL 리더보드 파일 경로 |
| `metric` | str | "proxy_score" 또는 "full_score" (기본값: "proxy_score") |
| `top_k` | int | 상위 몇 개를 반환할지 (기본값: 5) |
| `min_rung` | int | 분석에 포함할 최소 rung 번호 (기본값: 0) |

### 절차

```
1. JSONL 파일 읽기 (last-write-wins: 같은 run_id+trial_id는 마지막 항목 유효)

2. 완료 상태 필터 (status == "done")

3. metric 기준 정렬 → 상위 top_k 선정

4. proxy↔full 상관 계산 (full_score 있는 항목 대상)
   - Spearman ρ 계산 (stdlib만 사용, scipy 없이)
   - ρ < 0.7 이면 경고 플래그 세팅

5. 결과 딕셔너리 조립
   {
     "best": [trial 요약 목록],
     "spearman_rho": float,
     "proxy_trust": "high"/"medium"/"low",
     "anomalies": [주의 항목 목록]
   }

6. 반환
```

### 산출

```json
{
  "best": [
    {
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
      "cost": 16000
    }
  ],
  "spearman_rho": 0.82,
  "proxy_trust": "high",
  "anomalies": []
}
```

### 예시: 이상 감지 케이스

```json
{
  "best": [...],
  "spearman_rho": 0.51,
  "proxy_trust": "low",
  "anomalies": [
    "proxy top-1(t_003)이 full_score에서 rank-8로 하락 — 프록시 재검토 필요"
  ]
}
```

---

## 5. 두 Skill의 협력 관계

```
AutoResearch 루프
    │
    ├── [trial 생성]
    │
    ├── validity-gate ──────────► 무효 config 조기 차단
    │       │                       (GPU 낭비 방지)
    │       │ 유효
    │       ▼
    ├── [학습 실행 + proxy 계산]
    │
    ├── [leaderboard_write MCP 호출]
    │
    └── leaderboard-analysis ───► best 선정 + 이상 감지
                                    (의사결정 지원)
```

validity-gate는 **사전 필터**(비용 절감),
leaderboard-analysis는 **사후 분석**(의사결정 지원) 역할입니다.

---

## 6. 자신만의 Skill 설계하기 (확장 방법)

새 Skill을 추가하려면:

```bash
mkdir -p skills/my-skill
# skills/my-skill/SKILL.md 작성
```

SKILL.md에는 반드시 포함해야 할 섹션:
- **목적**: 1~2문장, 이 Skill이 없으면 에이전트가 무엇을 못 하는가
- **트리거 조건**: 어떤 상황에서 에이전트가 이 Skill을 호출해야 하는가
- **입력/산출**: 명확한 타입과 설명
- **절차**: 번호 매긴 단계, 분기 조건 명시

---

## 7. 체크리스트

- [ ] Skill이 시스템 프롬프트와 다른 점을 설명할 수 있다.
- [ ] `validity-gate`의 4가지 검사 단계를 순서대로 말할 수 있다.
- [ ] `leaderboard-analysis`가 proxy_trust를 "low"로 판정하는 조건을 설명할 수 있다.
- [ ] 두 Skill이 AutoResearch 루프에서 각각 어느 위치에 호출되는지 말할 수 있다.
- [ ] `skills/` 디렉터리에 새 Skill을 추가하는 방법을 안다.

---

## 관련 문서

- [00. 큰그림 · 진행순서](./00-overview.md)
- [01. Pi Agent 기초](./01-pi-agent-basics.md)
- [03. MCP 연결](./03-mcp.md)
- [06. 시스템 구조](./06-architecture.md)
- [08. 용어집](./08-glossary.md)
