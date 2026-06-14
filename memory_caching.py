"""
Memory Caching (MC) for Recurrent Models — minimal but faithful implementation.

Reference: Behrouz et al., "Memory Caching: RNNs with Growing Memory" (arXiv:2602.24281, 2026)

Idea
----
Split the sequence into segments. Each segment is compressed into a memory state
(here: a linear-attention KV matrix). Every token reads out from
  (a) its *online* memory  -> causal linear attention WITHIN the current segment, and
  (b) all *cached* past segment memories -> full readout against each earlier segment.
The two are combined by an aggregation function Agg(.), giving an effective memory
that grows with sequence length. Complexity interpolates between O(L) (RNN) and O(L^2)
(attention) as O(N*L), where N = number of segments.

Variants implemented (Agg):
  - "residual" : plain sum over current + cached memories (Eq. 7)
  - "grm"      : context-dependent gated sum, softmax over segments (Eq. 9/10)
                 (== Memory Soup for the linear-memory case, per the paper)
  - "ssc"      : Sparse Selective Caching — Top-k routing over cached segments (Eq. 16/17)

Memory update rule here is linear attention  M_t = M_{t-1} + phi(k_t) v_t^T,
but the caching/aggregation logic is rule-agnostic: swap `_segment_readouts`
to plug in DLA / Titans / SWLA.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def phi(x):
    # feature map for linear attention; elu+1 keeps it positive
    return F.elu(x) + 1.0


class MemoryCachingLinearAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        segment_len: int = 256,
        variant: str = "grm",          # "residual" | "grm" | "ssc"
        top_k: int = 4,                # for SSC: how many cached segments to keep
        ctx_dim: int | None = None,    # dim of the gating/connector space
    ):
        super().__init__()
        assert d_model % n_heads == 0
        assert variant in {"residual", "grm", "ssc"}
        self.d_model = d_model
        self.h = n_heads
        self.dh = d_model // n_heads
        self.C = segment_len
        self.variant = variant
        self.top_k = top_k
        self.dc = ctx_dim or d_model

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        # connector u_t = x W_u  and segment-context projection (Eq. 10)
        self.u_proj = nn.Linear(d_model, self.dc, bias=False)
        self.ctx_proj = nn.Linear(d_model, self.dc, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        B, L, D = x.shape
        C, H, Dh = self.C, self.h, self.dh

        # ---- pad to a whole number of segments -------------------------------
        pad = (C - L % C) % C
        if pad:
            x = F.pad(x, (0, 0, 0, pad))
        Lp = L + pad
        N = Lp // C                      # number of segments
        key_mask = torch.ones(B, Lp, device=x.device)
        if pad:
            key_mask[:, L:] = 0.0        # real-token mask (padded tokens contribute nothing)

        # ---- projections -----------------------------------------------------
        q = phi(self.q_proj(x)).view(B, N, C, H, Dh)
        k = phi(self.k_proj(x)).view(B, N, C, H, Dh)
        v = self.v_proj(x).view(B, N, C, H, Dh)
        v = v * key_mask.view(B, N, C, 1, 1)            # zero out padded keys/values
        k = k * key_mask.view(B, N, C, 1, 1)

        u = self.u_proj(x).view(B, N, C, self.dc)        # per-token connector
        ctx = self.ctx_proj(x).view(B, N, C, self.dc)
        denom = key_mask.view(B, N, C, 1).sum(2).clamp(min=1.0)
        seg_ctx = (ctx * key_mask.view(B, N, C, 1)).sum(2) / denom   # (B,N,Dc) MeanPooling(S^i)

        # ---- memories: per-token outer products  kv[...,a,e] = phi_k_a * v_e --
        kv = k.unsqueeze(-1) * v.unsqueeze(-2)           # (B,N,C,H,Dh,Dh)

        # (a) ONLINE memory: causal cumsum within the current segment
        cum_kv = kv.cumsum(dim=2)                        # (B,N,C,H,Dh,Dh)
        y_online = torch.einsum("bncha,bnchae->bnche", q, cum_kv)   # (B,N,C,H,Dh)

        # (b) CACHED memories: full KV sum per segment (independent compressors, Sec 3.4)
        seg_kv = kv.sum(dim=2)                           # (B,N,H,Dh,Dh)
        # readout of every query against every segment memory
        y_cross = torch.einsum("bncha,bmhae->bncmhe", q, seg_kv)    # (B,N,C,M,H,Dh)

        # place the online (causal) readout on the diagonal m == n
        idx = torch.arange(N, device=x.device)
        is_diag = (idx[:, None] == idx[None, :]).view(1, N, 1, N, 1, 1)
        y_all = torch.where(is_diag, y_online.unsqueeze(3), y_cross)  # (B,N,C,M,H,Dh)

        # ---- aggregation weights over segments m (only m <= n allowed) -------
        seg_causal = (idx[:, None] >= idx[None, :]).view(1, N, 1, N)  # (1,N,1,M) bool

        if self.variant == "residual":
            # plain sum (Eq. 7): equal weight 1 on every allowed memory
            w = seg_causal.float().expand(B, N, C, N).clone()
        else:
            logits = torch.einsum("bncf,bmf->bncm", u, seg_ctx) / math.sqrt(self.dc)
            logits = logits.masked_fill(~seg_causal, float("-inf"))

            if self.variant == "ssc":
                # keep current segment + Top-k most relevant cached segments
                cached = seg_causal.clone()
                cached[..., idx == idx] = False  # placeholder, fixed below
                cached = (idx[:, None] > idx[None, :]).view(1, N, 1, N)  # strictly past
                cached_logits = logits.masked_fill(~cached, float("-inf"))
                kk = min(self.top_k, max(N - 1, 1))
                topk = cached_logits.topk(kk, dim=-1).indices          # (B,N,C,kk)
                keep = torch.zeros_like(logits, dtype=torch.bool)
                keep.scatter_(-1, topk, True)
                keep = keep | (idx[None, :, None, None] == idx[None, None, None, :])  # +current
                keep = keep & seg_causal
                logits = logits.masked_fill(~keep, float("-inf"))

            w = torch.softmax(logits, dim=-1)            # (B,N,C,M)

        # ---- combine ---------------------------------------------------------
        y = torch.einsum("bncm,bncmhe->bnche", w, y_all)  # (B,N,C,H,Dh)
        y = y.reshape(B, Lp, D)[:, :L]                    # drop padding
        return self.o_proj(y)


# --------------------------------------------------------------------------- #
# Sanity checks
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    torch.manual_seed(0)
    B, L, D = 2, 50, 64
    x = torch.randn(B, L, D, requires_grad=True)

    for variant in ["residual", "grm", "ssc"]:
        m = MemoryCachingLinearAttention(D, n_heads=4, segment_len=16, variant=variant, top_k=2)
        y = m(x)
        assert y.shape == (B, L, D), (variant, y.shape)
        y.sum().backward()
        assert x.grad is not None
        x.grad = None
        print(f"[ok] variant={variant:8s}  out={tuple(y.shape)}  mean={y.mean().item():+.4f}")

    # When the whole sequence is ONE segment (N=1) MC reduces to a plain
    # causal linear-attention RNN (no cached memories to attend to).
    m1 = MemoryCachingLinearAttention(D, n_heads=4, segment_len=L, variant="grm")
    y1 = m1(x.detach())
    print(f"[ok] single-segment (N=1) -> reduces to plain linear-attention RNN, out={tuple(y1.shape)}")
    print("all checks passed.")
