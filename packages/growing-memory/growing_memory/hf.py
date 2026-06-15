"""HuggingFace transformers 호환 래퍼 — Unsloth Studio / HF Trainer / PEFT에서 로드·학습.

GrowingMemoryConfig(PretrainedConfig) + GrowingMemoryForCausalLM(PreTrainedModel)를 제공한다.
학습은 청크 병렬 O(L) forward를 사용(긴 컨텍스트 OOM 회피). transformers 설치 시에만 임포트됨.

주의(정직): Unsloth의 자동 커널 패치는 지원 아키텍처(Llama/Mistral 등) 대상이다. 본 커스텀
아키텍처는 Unsloth/HF '환경'에서 표준 Trainer·PEFT(LoRA)로 학습 가능하나, Unsloth 전용
fused 커널이 자동 적용되진 않는다. (대신 본 구현 자체가 O(L) 학습·O(1) 추론을 제공.)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutput

from .model import GrowingMemoryModel


class GrowingMemoryConfig(PretrainedConfig):
    model_type = "growing_memory"

    def __init__(self, vocab_size: int = 64, d_model: int = 256, n_layers: int = 4,
                 n_heads: int = 8, base_rule: str = "dla", aggregation: str = "residual",
                 segmentation: str = "logarithmic", init_mode: str = "independent",
                 segment_len: int = 64, top_k: int = 4,
                 ssl_encoder: str = "none", invariance_coeff: str = "low",
                 max_position_embeddings: int = 4096, train_chunk: int = 128, **kw):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.base_rule = base_rule
        self.aggregation = aggregation
        self.segmentation = segmentation
        self.init_mode = init_mode
        self.segment_len = segment_len
        self.top_k = top_k
        self.ssl_encoder = ssl_encoder
        self.invariance_coeff = invariance_coeff
        self.max_position_embeddings = max_position_embeddings
        self.train_chunk = train_chunk
        super().__init__(**kw)

    def to_core_cfg(self) -> dict:
        return {
            "base_rule": self.base_rule, "aggregation": self.aggregation,
            "segmentation": self.segmentation, "init_mode": self.init_mode,
            "segment_len": self.segment_len, "d_model": self.d_model,
            "n_layers": self.n_layers, "n_heads": self.n_heads, "top_k": self.top_k,
            "ssl": {"encoder": self.ssl_encoder, "invariance_coeff": self.invariance_coeff},
        }


class GrowingMemoryForCausalLM(PreTrainedModel):
    config_class = GrowingMemoryConfig
    supports_gradient_checkpointing = True

    def __init__(self, config: GrowingMemoryConfig):
        super().__init__(config)
        self.core = GrowingMemoryModel(config.to_core_cfg(), config.vocab_size,
                                       config.max_position_embeddings)
        self.train_chunk = config.train_chunk
        self.post_init()

    def get_input_embeddings(self):
        return self.core.front.emb

    def set_input_embeddings(self, new):
        self.core.front.emb = new

    def forward(self, input_ids=None, labels=None, attention_mask=None,
                use_cache=False, **kw):
        # 학습/prefill: 청크 병렬 O(L). (HF Trainer/Unsloth SFT 호환)
        logits = self.core.forward_chunked(input_ids, chunk=self.train_chunk)
        loss = None
        if labels is not None:
            sl = logits[:, :-1, :].contiguous()
            tl = labels[:, 1:].contiguous()
            loss = F.cross_entropy(sl.reshape(-1, sl.size(-1)), tl.reshape(-1),
                                   ignore_index=-100)
        return CausalLMOutput(loss=loss, logits=logits)

    @torch.no_grad()
    def generate_stream(self, input_ids, max_new_tokens: int = 64):
        """상수메모리 토큰 스트리밍 생성(O(1) 상태). 엣지 추론 데모."""
        self.eval()
        B = input_ids.shape[0]
        states = self.core.init_states(B, input_ids.device, self.core.head.weight.dtype)
        out = []
        t = 0
        for i in range(input_ids.shape[1]):           # prefill
            logits, states = self.core.step(input_ids[:, i], t, states); t += 1
        nxt = logits.argmax(-1)
        for _ in range(max_new_tokens):
            out.append(nxt)
            logits, states = self.core.step(nxt, t, states); t += 1
            nxt = logits.argmax(-1)
        return torch.stack(out, dim=1) if out else input_ids[:, :0]
