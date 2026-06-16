# 포지셔닝 & 로드맵 — 엣지학습노드 autoresearch

> 범용 옵티마이저가 목표가 아니다. **엣지 배포·온디바이스 적응 제약 하에서 (표현→기억) 구성을
> 단일 GPU로 자동 탐색하고, 엣지로 닫는** 전용 autoresearch 노드로 성장한다.

## 논문 대비 위치 (차이점)
새 알고리즘이 아니라 **통합·제약·실측**이 기여:
- karpathy/autoresearch(에이전트 ratchet) → **검증 게이트 + ASHA + 엣지 배포 제약(O(1) 추론)** 추가.
- Titans/fla(메모리 알고리즘) → **재구현 아님, vetted 채택** + autoresearch 스윕 + 효율↔정확도 실측.
- FlashAttention(메모리만 O(L)) → **연산까지 O(L)**(deltanet) 단일 4090 실측.
- 논문 일반(대형 클러스터) → **온프레미스 1 GPU + 엣지까지 닫힌 루프**.

## 실측 범위 (정직)
- ✅ 견고: 병렬↔재귀 동치 1e-7, chunked O(L), 4090 128K 학습, 연산 O(L) vs O(L²)(8×@128K),
  추론 KV O(L) vs 상태 O(1)(~992×), vetted recall 1.0(deltanet/titans), ASHA 조기중단.
- 🟡 제한: 합성 MQAR·초소형 모델·소예산·단일시드. HC-SR04는 실데이터지만 900행.
- ❌ 미실측: 실 스케일 데이터·1.3B+·downstream full·물리 엣지 배포.
→ *proof-of-mechanism*까지. 실응용은 아래 로드맵으로.

## 로드맵
```
[엣지 센서]→수집→[NAS]→샤드→[4090 AutoResearch]→best(O(1) 상태) export→[엣지 재배포]
   ↑________________________ 지속 루프(엣지가 데이터 생성) ________________________|
```
- **T1 (착수→1차 완료)**: 실센서(HC-SR04) 데이터를 autoresearch 학습 루프에 연결 — 토큰화→다음값
  예측 자기지도. vetted base_rule 실데이터 학습. (`t1_hcsr04.py`, 4090)
  - **결과**: deltanet **0.983**, retention 0.982, gla 0.975, linear 0.95 vs copy-baseline **0.934**
    → 실데이터가 루프에 흐르고 모델이 baseline 초과(deltanet 최고, 메모리 결과와 일관).
  - **정직 단서**: 과제가 쉬움(9 상수블록→copy 0.93, 900토큰). 시간순 분할은 분포이동 버그(모델
    0.05까지 추락)→전구간 윈도우로 수정. **실가치 과제는 이상/드리프트 탐지**(HCSR04.md: −22.8 SE).
  - **교훈**: 실센서는 구조에 맞는 과제 설계 필요(다음값 LM은 퇴화) + 시계열은 분포이동 분할 주의.
- **T2**: 스케일·통계 — 300M~1.3B, 다중시드, downstream full, 실예산 ASHA.
- **T3**: 물리 엣지 배포 — best(deltanet/titans, O(1) 상태)를 Pi/Jetson 실장비 상수메모리 스트리밍.
- **T4**: 온디바이스 적응 — Titans 테스트시점 메모리로 엣지가 추론+적응, autoresearch가 "엣지 적응형
  config"를 탐색 = 진짜 엣지학습노드.

## 차별점 (밀고 갈 것)
엣지 제약 인식 autoresearch(O(1) 추론 + 온디바이스 적응을 *제약*으로) · 온프레미스 단일 GPU ·
paper-faithful 검증 게이트 · Pi 에이전트 + 통합 Web/MCP/도커/엣지.
