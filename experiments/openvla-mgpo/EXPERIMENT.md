# EXPERIMENT (사전등록) — MGPO + compression-coverage를 OpenVLA에 적용해 물리오류 행동 감소

> 사전등록(open-research): *데이터를 보기 전에* 가설·지표·합격선을 고정한다. 사후에 합격선을 바꿔
> PASS로 만들지 않는다. 합격선 없는 라운드는 INCONCLUSIVE.

## 배경 / 참조
- **OpenVLA**(로봇 조작 VLA): 행동을 **이산 토큰**으로 출력 → 자기회귀 토큰 정책.
- **VibeThinker**(WeiboAI, arXiv 2511.06221): **MGPO**(MaxEnt-Guided Policy Optimization) — 정책이
  가장 불확실한(고엔트로피) 문제를 우선해 on-policy 학습, 정답 신호 증폭 + 다양성 유지(mode-collapse 방지).
  **Compression-Coverage 가설**: *verifiable* 능력은 parameter-dense → 소형 코어로 압축 가능.
- **검증가능 보상**: LIBERO 조작 과제의 성공/실패(+물리타당) = verifiable reward → MGPO 전제 충족.

## 핵심 가설
1. **H1 (MGPO→물리오류↓)**: LIBERO 성공/물리타당을 verifiable reward로 MGPO(LoRA) RL하면, OpenVLA의
   *물리오류 행동률*이 baseline 대비 **유의하게 감소**한다.
2. **H2 (다양성 유지)**: MGPO는 다양성을 유지해(엔트로피), 단순 RL(mode-collapse) 대비 미지 과제 일반화가
   낫다.
3. **H3 (compression-coverage)**: 물리정합성은 parameter-dense → MGPO로 얻은 정책을 **소형 코어**(LoRA/
   소형 VLA)로 압축해도 물리오류 감소가 대부분 유지된다(엣지 배포 가능).

## 설계
- 환경/보상: **LIBERO**(OpenVLA 표준). reward = task success(이진) + (가능시) 물리타당 패널티.
- 정책: **OpenVLA-7b** + **LoRA**(단일 4090 24GB 제약). 토큰 정책 → MGPO 직접 적용.
- 방법: S0 baseline → S1 MGPO LoRA → S2 compression(소형 코어) → S3 autoresearch config 스윕.

## 지표
- **success rate**(주지표, ↑), **physics-error rate**(물리위반/실패행동률, ↓; LIBERO 실패+충돌/관절한계 proxy),
- 출력 **엔트로피/다양성**(다양성 유지 점검), 미지 과제 일반화 gap.

## 합격선 (사전 고정)
- **H1 PASS**: physics-error rate가 baseline 대비 **상대 ≥20% 감소** AND success rate 비열화(≥ baseline−1%p),
  동일 LIBERO 스위트/시드에서.
- **H2 PASS**: 출력 엔트로피가 단순 RL 대비 높음 + 미지 과제 success 우위.
- **H3 PASS**: 압축 코어가 H1 감소폭의 **≥80% 유지**, 추론 메모리 O(1)/소형.
- 합격선 미정/미측정 → INCONCLUSIVE.

## 자원 / 실현성
- 1× RTX 4090 24GB(OpenVLA-7b 가중치 캐시됨). **fp16 로드(≈14GB)**. full RL 불가 → **LoRA + 시뮬 rollout**.
- LIBERO/robosuite/mujoco 설치 필요(S0). multi-day 연구 프로그램.

## 단계
- **S0(현재)**: OpenVLA-7b 로드·행동예측 확인 + LIBERO baseline(success/physics-error) 측정 → verifiable 신호 확립.
- **S1**: MGPO LoRA RL → H1/H2 검증.
- **S2**: compression-coverage → H3.
- **S3**: autoresearch 노드에 MGPO config 스윕(보상=success) 연결 + 엣지 배포.

## 진행 로그
### S0a — 정책 측 확인 (완료, 4090)
- 환경: 전용 venv `~/openvla_venv` (transformers==4.40.1, timm==0.9.10, accelerate; torch 2.10/cu128).
  gm_venv(transformers 5.12)는 OpenVLA 원격코드와 비호환(`AutoModelForVision2Seq` 제거됨) → 전용 venv 필요.
- 결과: **OpenVLA-7b fp16 로드 성공 (7.54B params, peak 15.1GB / 24GB), `predict_action` → 7-DoF action 출력 OK**
  (`s0_load_openvla.py`, unnorm_key=bridge_orig). 검증 루프의 *정책 측* 확립.
- 다음(S0b): LIBERO 설치 → 시뮬 rollout으로 success/physics-error **baseline** 측정(*보상 측*).

### S0b — LIBERO baseline 측정 (완료, 4090)
- 환경: LIBERO + robosuite 1.4.1 + mujoco 3.9.0, EGL 헤드리스 렌더(MUJOCO_GL=egl). 모델
  `openvla/openvla-7b-finetuned-libero-spatial` fp16. 하니스 `s0b_libero_baseline.py`.
- **버그→수정(정직)**: 첫 런 성공률 0% → 원인은 그리퍼 부호 규약(OpenVLA는 0=close/1=open, LIBERO는
  -1=open/+1=close) 누락. 공식 eval과 동일하게 `normalize_gripper_action(binarize)+invert` 추가 → 0%→66%.
- **결과 (libero_spatial, 10 task × 5 ep = 50 episodes, max 220 step, ~22분)**:
  - **success_rate = 0.66** (verifiable reward 신호 확립), failure_rate = 0.34
  - physics proxy: **joint_limit_rate = 0.17%**(관절한계 근접), **mean_jerk = 0.186**(행동 급변), mean_trans_norm = 0.65
- **정직 단서**: 66%는 논문 보고치(spatial ~84%)보다 낮음 — 부분측정(50ep·init-state 5개·우리 시드,
  논문은 500ep)이라 분산이 큼. "부분 기준선"으로 사용. full-suite(10×50) 재측정은 별도.
- 이 수치가 **S1 MGPO가 개선해야 할 기준선**: H1 = physics-error proxy(joint-limit/jerk) 상대 ≥20%↓
  AND success ≥ 0.66−1%p.

### S0b 정성 분석 — 어떤 태스크가 왜 실패했나 (실패 영상 프레임 직접 확인)
태스크별 성공률 (libero_spatial, 10 task × 5 ep):

| 난이도 | task | 성공 | 평균 step | jerk | 관절한계율 |
|---|---|---|---|---|---|
| 어려움 | next_to_the_plate | 1/5 | 199 | 0.225 | 0% |
| 어려움 | on_the_ramekin | 1/5 | 196 | 0.225 | 0% |
| 어려움 | on_the_wooden_cabinet | 2/5 | 184 | 0.220 | 0% |
| 보통 | in_the_top_drawer | 3/5 | 168 | 0.192 | 1.66% |
| 보통 | next_to_the_ramekin | 3/5 | 156 | 0.143 | 0% |
| 보통 | from_table_center | 4/5 | 152 | 0.162 | 0.05% |
| 보통 | on_the_stove | 4/5 | 157 | 0.156 | 0% |
| 쉬움 | between_the_plate_and_the_ramekin | 5/5 | 83 | 0.209 | 0% |
| 쉬움 | next_to_the_cookie_box | 5/5 | 100 | 0.156 | 0% |
| 쉬움 | on_the_cookie_box | 5/5 | 89 | 0.174 | 0% |

**관찰 (정성)**:
1. **실패의 본질 = 물리 난폭함이 아니라 공간 그라운딩 실패.** 실패 에피소드도 jerk(0.10~0.24)·관절한계율
   (~0%)이 정상. 팔이 거칠어서가 아니라 **엉뚱한 위치로 가 그릇을 못 집고 220스텝을 헛돈다**(얌전한 실패).
2. **어려운 태스크 = 참조 표현이 모호/타겟이 가려짐.** 최악 3개(next_to_the_plate, on_the_ramekin,
   on_the_wooden_cabinet)는 같은 검은 그릇이 여러 개라 *어느 그릇인지(참조 해소)*가 핵심. 쉬운 것은 위치가 명확.
3. **관측 실패 모드**: 팔이 타겟이 아닌 테이블 좌·중앙으로 표류 → 그리퍼를 연 채 낮게 떠 정지 → 그랩 미시도.
   (프레임 증거: report-srv `/frames/`의 t1_ep0_FAIL, t4_ep1_FAIL 시작/중반/종료)
4. **실험 함의**: baseline 실패가 physics proxy로 거의 설명되지 않음 → MGPO **물리보상의 개선 여지가 제한적**일 수
   있고, ARCHITECTURE.md ① referential/grounding 축(에이전트·object-memory)이 성공률에 더 직접적. H1 검증 시
   physics-error 감소와 **success 변화**를 반드시 함께 보고, grounding 축 개입(Phase 1)도 비교군에 포함한다.

## 정직 단서
- MGPO는 검증가능 추론(math/code)에서 입증됨 — VLA 물리행동 전이는 **본 실험의 가설**(미입증).
- 결과는 소형(LoRA)·단일GPU 범위; 대규모/실로봇 일반화는 별도.
