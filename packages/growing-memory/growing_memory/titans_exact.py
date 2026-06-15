"""titans_exact — 논문(arXiv:2501.00663) 식을 직접 옮긴 *교육용 참조*. ⚠️ 학습 안 됨(chance).

⚠️ 정직 경고: 이 from-scratch 구현은 MQAR 회상에서 **chance 수준(≈0.06)으로 학습되지 않는다**
(실측: 안정화 후에도 recall 0.064, loss=ln(16) 정체). 논문의 실제 성능을 내려면 청크 병렬화·
정확한 기울기·메모리 MLP·세심한 정규화 등이 필요하며, 본 단순 재현은 그걸 충족하지 못한다.
**실사용/정확 구현은 lucidrains `titans-pytorch`(검증: recall 0.98)를 쓰라** → `growing_memory.titans_backend`.
이 파일은 식의 형태를 보여주는 교육 목적으로만 남긴다.

식(arXiv:2501.00663, Behrouz et al., Google):

기존 model.py의 base_rule="titans"는 단순화 근사(dla 감쇠 + 정적 영속읽기)였다. 이 모듈은
논문의 **테스트시점 경사 갱신**을 충실히 구현한다(선형/행렬 메모리 변형):

  ℓ(W; x_t) = ||W·k_t − v_t||²,         k_t=x_tW_K, v_t=x_tW_V, q_t=x_tW_Q
  ∇ℓ        = (W·k_t − v_t)·k_tᵀ        (행렬 메모리의 정확 기울기)
  S_t       = η_t·S_{t−1} − θ_t·∇ℓ       (모멘텀: 과거 surprise + 현재 surprise)
  W_t       = (1 − α_t)·W_{t−1} + S_t    (α_t: 데이터 의존 망각/weight-decay 게이트)
  y_t       = W_t·q_t                    (현재 메모리로 읽기)

η_t(모멘텀), θ_t(학습률), α_t(망각)는 데이터 의존 스칼라(입력에서 예측). 추론 상태는 (W,S)
두 행렬 = O(d²) 상수 → 길이 무관(상수메모리 추론). 학습은 토큰 재귀(autograd)로 정확.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralMemoryExact(nn.Module):
    """논문 식 그대로의 선형 신경 장기기억. [B,L,d] → [B,L,d]. 상태 (W,S) ∈ R^{d×d}."""

    def __init__(self, dim: int, n_persist: int = 4):
        super().__init__()
        self.d = dim
        self.WK = nn.Linear(dim, dim, bias=False)
        self.WV = nn.Linear(dim, dim, bias=False)
        self.WQ = nn.Linear(dim, dim, bias=False)
        # 데이터 의존 게이트: η(모멘텀), θ(학습률), α(망각)
        self.gate = nn.Linear(dim, 3)
        nn.init.zeros_(self.gate.bias)
        # 영속(persistent) 메모리 — 입력 독립 학습 파라미터(논문)
        self.persist = nn.Parameter(torch.zeros(n_persist, dim))
        nn.init.normal_(self.persist, std=0.02)
        self.pq = nn.Linear(dim, dim, bias=False)
        self.onorm = nn.LayerNorm(dim)                         # 읽기 출력 정규화(안정화)

    def forward(self, x):
        B, L, d = x.shape
        k = self.WK(x); v = self.WV(x); q = self.WQ(x)         # [B,L,d]
        # 키/쿼리 L2 정규화 → 기울기/읽기 스케일 안정화(메모리 폭주 방지)
        k = F.normalize(k, dim=-1)
        q = F.normalize(q, dim=-1)
        g = self.gate(x)                                       # [B,L,3]
        eta = torch.sigmoid(g[..., 0])                         # 모멘텀 ∈(0,1)
        theta = F.softplus(g[..., 1]) * 0.1                    # 학습률 >0 (스케일 안정화)
        alpha = torch.sigmoid(g[..., 2])                       # 망각 ∈(0,1)

        W = x.new_zeros(B, d, d)
        S = x.new_zeros(B, d, d)
        ys = []
        for t in range(L):
            kt, vt, qt = k[:, t], v[:, t], q[:, t]             # [B,d]
            Wk = torch.bmm(W, kt.unsqueeze(-1)).squeeze(-1)    # W·k_t [B,d]
            grad = (Wk - vt).unsqueeze(-1) * kt.unsqueeze(1)   # (W·k−v)·kᵀ [B,d,d]
            S = eta[:, t, None, None] * S - theta[:, t, None, None] * grad
            W = (1.0 - alpha[:, t, None, None]) * W + S
            yt = torch.bmm(W, qt.unsqueeze(-1)).squeeze(-1)    # 읽기 [B,d]
            ys.append(yt)
        y = torch.stack(ys, dim=1)                            # [B,L,d]
        y = self.onorm(y)                                     # 읽기 출력 정규화
        # 영속 메모리 읽기(정적): softmax attention to learnable tokens
        pq = self.pq(x)                                        # [B,L,d]
        ps = (pq @ self.persist.t()) / (d ** 0.5)             # [B,L,P]
        y = y + ps.softmax(-1) @ self.persist                 # [B,L,d]
        return y

    def state_bytes(self, batch: int) -> int:
        return batch * 2 * self.d * self.d * 4                 # W + S, float32


class TitansExactLM(nn.Module):
    """검증용 최소 LM: emb+pos → [norm→NeuralMemoryExact→res → norm→MLP→res]×L → head."""

    def __init__(self, vocab: int, dim: int = 128, depth: int = 2, max_len: int = 1024):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        self.pos = nn.Parameter(torch.zeros(1, max_len, dim)); nn.init.normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(nn.ModuleDict({
                "n1": nn.LayerNorm(dim), "mem": NeuralMemoryExact(dim),
                "n2": nn.LayerNorm(dim),
                "mlp": nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim)),
            }))
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab, bias=False)

    def forward(self, idx):
        h = self.emb(idx) + self.pos[:, :idx.shape[1]]
        for b in self.blocks:
            h = h + b["mem"](b["n1"](h))
            h = h + b["mlp"](b["n2"](h))
        return self.head(self.norm(h))
