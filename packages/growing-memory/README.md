# growing-memory-pytorch

설계 4축(`base_rule`: linear/swla/dla/titans · `aggregation`: residual/grm/soup/ssc ·
`segmentation` · `init_mode`) + SSL 표현축의 PyTorch 구현.

- **학습/prefill**: 청크 병렬 **O(L)** (`forward_chunked`) — 긴 컨텍스트 OOM 회피(단일 4090 128K).
- **추론**: 재귀 **O(1) 상태**(`forward_recurrent` / 토큰 `step`/`stream`) — KV캐시 없이 상수 메모리.
- 세 경로(naive/chunked/recurrent) 출력 동치(~1e-7).

## 설치
```bash
pip install -e packages/growing-memory          # 코어(torch)
pip install -e "packages/growing-memory[hf]"    # + HuggingFace/Unsloth 래퍼(transformers)
```

## 1) 코어 사용
```python
import growing_memory as gm
cfg = {"base_rule":"titans","aggregation":"residual","segmentation":"logarithmic",
       "init_mode":"independent","segment_len":64,"d_model":256,"n_layers":4,"n_heads":8,
       "ssl":{"encoder":"none","invariance_coeff":"low"}}
model = gm.build_model(cfg, vocab=64, max_len=4096)
gm.run_equivalence_test(cfg)        # 청크/재귀 동치 검증(유효성 게이트) → True/False
run = gm.RealRun(cfg, {"device":"cuda","task":"factory_mqar","vocab":64,
                       "seq_len":512,"num_pairs":4,"batch":32,"lr":3e-3})
run.train(2000); print(run.proxy_score("factory_mqar", None))
```

## 2) AutoResearch 노드 연결
`model_adapter`가 `import growing_memory` 후 `build()` / `run_equivalence_test()` / `RealRun`을
자동 사용한다(설치만 하면 reference 구현 대신 이 패키지가 실물 백엔드). `GROWING_MEMORY_HOME`도 지원.

## 3) HuggingFace / Unsloth Studio
```python
from growing_memory.hf import GrowingMemoryConfig, GrowingMemoryForCausalLM
cfg = GrowingMemoryConfig(vocab_size=32000, d_model=512, n_layers=8, base_rule="titans")
model = GrowingMemoryForCausalLM(cfg)          # transformers PreTrainedModel
# HF Trainer / TRL SFTTrainer / PEFT(LoRA)로 학습 가능 (input_ids/labels)
out = model(input_ids=ids, labels=ids)          # out.loss, out.logits
gen = model.generate_stream(ids, max_new_tokens=64)   # O(1) 상수메모리 스트리밍 생성
```
> 정직 단서: Unsloth의 *자동 fused 커널*은 지원 아키텍처(Llama 등) 대상이라 본 커스텀 아키텍처엔
> 자동 적용되지 않는다. 대신 Unsloth/HF 환경에서 표준 Trainer·LoRA로 학습되며, 본 구현 자체가
> O(L) 학습·O(1) 추론을 제공한다.

## 라이선스
Apache-2.0
