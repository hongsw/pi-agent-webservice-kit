# 아키텍처 — object-centric temporal state memory for OpenVLA

## 문제 정의 (두 가지 오류축)
OpenVLA(DINOv2+SigLIP → MLP projector → LLaMA-2 7B → action head)는 "현재 이미지 + 현재 명령" 중심.
따라서 두 종류의 오류가 난다 — **이 둘을 분리해 다른 방법으로 푼다**:
1. **Referential / object-permanence 오류**: "빨간 공을 잡아라 → 그거를 던져라"의 *그거*가 이전 객체로
   resolve 안 됨. → **상태 메모리/에이전트 문제** (이번 문서).
2. **물리(physics) 오류**: 충돌·관절한계·드롭 등 물리적으로 틀린 행동. → **MGPO 검증보상 RL 문제**
   ([EXPERIMENT.md](EXPERIMENT.md)).

핵심 판단: **"projector만 키우면 부족"**. 기억은 projector의 역할이 아니다. perception→object state→
language grounding→action 사이에 **object-level memory path**를 새로 만들어야 한다.

## 이상적 in-model 스택 (Phase 2 목표)
```
RGB → DINOv2/SigLIP → Object Grounding+Tracking → Object State Memory
    → State-aware Projector (cross-attn / memory tokens) → LLaMA-2 VLA → Action Head(+skill-state)
```
object token 예: {object_id, category, color, bbox, mask, pose_3d, held_by_gripper, last_instruction_ref}.

## 단계 결정 (검증 먼저, 내장 나중)
- **Phase 1 — 에이전트/외부 메모리 (1차, 빠른 검증)**: OpenVLA **frozen = stateless executor**. 앞단에
  **DeepAgent = stateful planner**(object memory·referent resolver). 모델 안 건드리고 빠르게 검증.
- **Phase 2 — 모델 내장 (제품화/저지연/강건)**: object memory token + state-aware(cross-attn) projector +
  **LoRA**. → 여기서 **temporal state memory = 우리 growing-memory(deltanet/titans, O(1) 재귀 상태)**로 구현
  가능(상태가 객체 식별을 시간축으로 유지, 엣지 상수메모리). ← 이번 세션 작업과 직접 연결.

## Phase 1 구성 (DeepAgent 레이어)
```
User instruction → DeepAgent planner → object memory 조회/갱신 → referent resolver("그거"=red ball)
                 → OpenVLA action 호출(stateless) → 관찰결과 저장 → 반복
```
도구(tool) 분리:
- **Perception Tool**: detect(red ball) / track(object_id)  — Grounded-SAM / OWL-ViT / Detic / SAM + tracker
- **Memory Tool**: object_state 저장/갱신 (obj_1=red ball, grasped=true, task_phase=GRASPED)
- **Resolver Tool**: "그거" → last_referred_object 또는 gripper_contact_object
- **VLA Tool**: OpenVLA executor 호출

```python
class ObjectMemory:
    active_object_id; object_slots; last_referred_object; gripper_contact_object; task_phase
# 매 timestep: detect/track → id 갱신 → instruction referent 추출 → 그거 resolve →
#              memory token을 (Phase2) LLM 입력에 추가 → action head가 active_object_state 기반 예측
```

## 연결 프로토콜 (DeepAgent ↔ OpenVLA)
OpenVLA는 자연어 instruction + image만 받는 **stateless executor**. DeepAgent가 stateful planner로
referent를 resolve해 *구체 명령*으로 변환해 넣는다.
```
DeepAgent(stateful)  →  Task Resolver  →  OpenVLA(stateless): image + "throw the object held by the gripper" → 7-DoF
```
Task API:
```json
{ "task_name": "throw", "object_id": "obj_1", "object_label": "red ball",
  "state": { "grasped": "true", "active": "true" } }
```
→ OpenVLA 입력 instruction으로 변환: "throw the red ball" 또는 "throw the object currently held by the gripper".

## 우선순위
1 Object State Memory · 2 Tracking/Persistence · 3 State-aware Projector · 4 Referential Resolver ·
5 Action Head/Skill-state(grasp→throw phase).

## 이번 실험과의 통합
- Phase 1(agent) = referential 오류 ↓ (빠른 검증, DeepAgent + frozen OpenVLA).
- MGPO RL(별도 축) = 물리 오류 ↓ (verifiable reward).
- Phase 2 temporal memory = **growing-memory(deltanet/titans) O(1) 재귀 상태** → 엣지 내장.
- compression-coverage = 객체상태+정책을 소형 코어로 압축 → 엣지학습노드.

다이어그램: [architecture.svg](architecture.svg)
