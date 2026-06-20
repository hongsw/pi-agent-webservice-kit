# EXPERIMENT (사전등록) — 트랙 C: VibeThinker-3B 재현 후 개선

> 사전등록(open-research): *데이터를 보기 전에* 무엇을 어떤 합격선으로 재현할지 고정한다.
> 1차 목표는 **재현(reproduction)** — 공개 주장(claim)을 우리 환경에서 직접 측정해 검증/반증한다.

## 배경 / 참조
- **VibeThinker-3B** (WeiboAI, MIT): Qwen2.5-Coder-3B 기반, SSP(Spectrum-to-Signal) + MGPO +
  Long2Short Math RL로 학습. 소형(3B)인데 수학/코딩에서 매우 높은 점수를 주장.
- 출처: HF `WeiboAI/VibeThinker-3B`, GitHub `github.com/WeiboAI/VibeThinker`,
  소개글 https://discuss.pytorch.kr/t/vibethinker-3b-3b-feat-weibo-ai/10748
- 트랙 B(OpenVLA×MGPO)와 **MGPO를 공유**하지만 도메인(수학 추론 vs VLA 물리)이 달라 **별개 트랙**.

## 검증 대상 주장 (공개 수치)
| 벤치 | claim (CLR 적용) | claim (CLR 미적용) |
|---|---|---|
| AIME25 | 96.7 | 91.4 |
| AIME26 | 97.1 | 94.3 |
| HMMT25 | 95.4 | — |
| LiveCodeBench v6 | 80.2 (Pass@1) | — |
| GPQA-Diamond | 72.9 | — |

> ⚠️ 정직 단서: 3B 모델의 AIME25 96.7은 프런티어급으로 **비정상적으로 높음**. 특히 **CLR**(Claim-Level
> Reliability)은 *test-time scaling*이며 **구현이 공개되지 않음**. 따라서 우리는 **CLR 미적용 단일추론**
> 재현을 1차 기준으로 삼고, CLR 수치는 직접 재현 불가로 분류한다.

## 핵심 질문 (재현)
- **R1 (로드/추론)**: VibeThinker-3B를 우리 4090에서 로드해 수학 문제에 사고연쇄+정답을 내는가?
- **R2 (AIME 부분재현)**: AIME25(30문항)에서 우리 측정 pass@1 / avg@k가 claim(CLR 미적용 91.4)에
  **근접**하는가? 큰 격차면 그 원인(샘플수·max_tokens·파서·프롬프트)을 규명.
- **R3 (효율/개선 여지)**: 재현 후, 추론 토큰/시간 대비 정확도 트레이드오프에서 개선 지점(예: Long2Short,
  CLR 대체 검증, 소형화)을 찾는다.

## 설계
- 모델: `WeiboAI/VibeThinker-3B` (bf16, 단일 RTX 4090 24GB — 공유 GPU).
- 추론 설정(공식): **temperature=1.0, top_p=0.95, top_k=-1**, max_new_tokens=40960(권장).
  1차 재현은 자원 절감 위해 max_tokens 축소(예: 16k~32k)부터, 토큰절단 영향 별도 기록.
- 프레임워크: 1차 transformers(gm_venv, tf 5.12) — 단순/통제. 대규모는 vLLM==0.10.1 별도 고려.
- 데이터: AIME 2025 (HF 공개 데이터셋, 30문항). 정답=정수 → 규칙기반 채점(LLM judge 불필요).

## 지표 / 채점
- **pass@1**(주지표): 그리디/단일샘플 정답률. **avg@k**(가능 시): k 샘플 평균 정답률(temp=1.0).
- 정답 추출: `\boxed{}` 우선, 없으면 마지막 정수. 정수 일치로 채점.
- 보조: 평균 생성 토큰 수, 문항당 추론 시간(효율).

## 합격선 (사전 고정)
- **R1 PASS**: 최소 1개 AIME 문항에서 사고연쇄 후 `\boxed{정답}` 생성.
- **R2 판정**: 우리 pass@1이 claim(91.4, CLR미적용)의 **±10%p 이내**면 "재현 성공";
  10~30%p 낮으면 "부분재현(설정차)"; 30%p+ 낮으면 "재현 실패(원인규명 필요)".
  (단일샘플 pass@1은 avg@64보다 낮게 나오는 게 정상 — 이 점을 감안해 판정.)
- 합격선 미측정 → INCONCLUSIVE.

## 단계
- **S0(현재)**: 가중치 다운로드 + R1(로드·단일문항 추론) — GPU 여유 시 실행.
- **S1**: AIME25 30문항 pass@1 (max_tokens 16k→32k) → R2 1차 판정.
- **S2**: avg@k(예 k=4~8) + 토큰/시간 프로파일 → 효율(R3) 분석.
- **S3**: 개선 방향 실험(Long2Short / 검증 기반 self-consistency 등).

## 자원 / 현실성
- 공유 4090(타 사용자 학습 시 6~8GB만 여유). 3B bf16 가중치 ≈6.5GB.
  긴 생성(40k 토큰) KV가 크므로, 여유 부족 시 GPU가 빌 때 자동 실행하도록 큐잉.

## 진행 로그
### S0 — 착수
- 가중치 `WeiboAI/VibeThinker-3B` 다운로드(gm_venv, transformers 5.12).

### R1 — 로드·단일문항 추론 (PASS, 4090)
- **GPU 경합**: 타 사용자(imshen19)가 OpenVLA 학습으로 17GB 점유 → 여유 6.9GB. bf16(~6.5GB+KV)이 안 들어가
  **8-bit 양자화로 sanity** 실행(재현 수치 아님, 파이프라인+추론력 확인용).
- **결과**: AIME 2024-I-1(정답 204)을 **정확히 해결**. 단계적 사고연쇄 후 `\boxed{204}` 생성, correct=True.
  생성 2225토큰, 8-bit 7.3 tok/s(303s, 느림 — 양자화 영향).
- **판정**: **R1 PASS** — 모델 로드+추론+정답추출 파이프라인 검증, VibeThinker 추론력 실측 확인.
- **다음(S1)**: bf16 정식 설정(temp=1.0, top_p=0.95)으로 AIME25 30문항 pass@1 → claim 91.4(CLR미적용) 대비.
  GPU 17GB 여유 시 실행(현재 큐잉). 8-bit는 느리고 정확도 영향 있어 벤치는 bf16로.
