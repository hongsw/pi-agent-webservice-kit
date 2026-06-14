# real/ — 실물 growing-memory PyTorch 참조 구현

업스트림 `growing-memory-pytorch`가 없을 때, 설계 4축 + SSL 표현축을 **실제 torch 모듈**로
실현하고 **실제 MQAR 연관회상 과제로 학습/평가**한다. 4090에서 실학습으로 동작.

> 백엔드 우선순위(`model_adapter.build`): 업스트림(GROWING_MEMORY_HOME) → **이 참조 구현(torch)** → mock.
> 강제 mock: `AR_FORCE_MOCK=1`. cuda 없으면 cpu 폴백.

## 설계 축 → 구현 매핑

| 축 | 후보 | 구현(model.py) |
|---|---|---|
| base_rule | linear | 정규화 선형 어텐션(feature map elu+1 + 분모 정규화) |
| | dla | + 데이터 의존 스칼라 감쇠 게이트(bias=4 → 초기 망각 최소) |
| | titans | dla + 영속(persistent) 메모리 softmax 읽기("memory as context") |
| | swla | 슬라이딩 윈도우 소프트맥스 어텐션(window=segment_len) |
| aggregation | residual | concat heads → Wo |
| | grm | head별 sigmoid 게이트 후 concat |
| | soup | head별 d_model 투영 후 평균(model-soup식) |
| | ssc | 상위 top_k head만 sigmoid 게이트(희소 선택) |
| segmentation | constant/logarithmic | 감쇠 floor(0.0 / 0.9) → 메모리 길이 조절 |
| init_mode | checkpoint/independent | 컨텍스트 평균 임베딩으로 메모리 초기화 여부 |
| SSL encoder | vjepa/dinov2/vicreg | 표현 aux loss(invariance_coeff로 가중) |

## 과제(data.py)
- **factory_mqar**: 컨텍스트 내 (key,value) 쌍을 흩뿌리고 끝에서 key로 value 회상 →
  유효 메모리 성장 가설을 직접 측정. recall(정확도)로 평가.
- **short_horizon_pred**: 주기적 반복+잡음 토큰열의 다음 토큰 예측(자기지도) → val CE/bpb.

## 학습/평가(trainer.py)
`RealRun`이 모델·AdamW·평가셋을 보유. ASHA 승급 시 같은 RealRun을 계속 학습(누적 스텝).
`proxy_score`=프록시 점수, `full_score`=더 어려운 설정의 평가.

## 알려진 한계 (실측)
- MQAR recall은 **grokking(급격 상전이)** 특성이라 학습이 임계 스텝에서 급증한다.
  현재 예산(rungs≤2500)에서 `linear`/`swla`+`residual/grm`은 chance 위로 학습되나,
  `dla`/`titans`/`ssc`/`soup`는 상전이 임계가 더 높아 같은 예산에서 chance 근처에 머문다.
  → 이는 선형/소프트맥스 어텐션이 정확 회상에 유리하다는 문헌(Zoology) 경향과 일치하나,
    일부는 더 긴 예산/하이퍼파라미터 튜닝이 필요한 구현 한계이기도 하다(추후 보정 대상).
- 실 데이터(공장 샤드) 미연결: MQAR은 합성이지만 실제 학습 신호. NAS 연결 시 교체.
