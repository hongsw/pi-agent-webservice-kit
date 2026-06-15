"""실물 GrowingMemoryModel — 설계 4축 + SSL 표현축의 PyTorch 참조 구현.

base_rule (시퀀스 믹싱/기억 규칙):
  - linear : 감쇠 없는 선형 어텐션(retention식, softmax-free). D = causal ones.
  - dla    : 데이터 의존 스칼라 감쇠 게이트(gated linear attention).
  - titans : dla 감쇠 + 영속(persistent) 메모리 읽기("memory as context").
  - swla   : 슬라이딩 윈도우 소프트맥스 어텐션(window=segment_len).
aggregation (head 결합):
  - residual / grm(게이트) / soup(평균) / ssc(top_k 선택).
segmentation: constant(감쇠 자유) vs logarithmic(감쇠 floor↑ → 긴 메모리).
init_mode: checkpoint(컨텍스트 평균 임베딩으로 메모리 초기화) vs independent.
SSL front: vjepa/dinov2/vicreg aux loss(invariance_coeff로 가중) — 표현축을 실제로 학습에 반영.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return self.w * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


# ── SSL 표현 front ───────────────────────────────────────────────────────────
class SSLFront(nn.Module):
    """임베딩 + 위치인코딩 + SSL aux loss(encoder 변형별)."""

    def __init__(self, vocab: int, d_model: int, max_len: int,
                 encoder: str, invariance_coeff: str):
        super().__init__()
        self.encoder = encoder
        self.coeff = {"low": 0.05, "high": 0.5}.get(invariance_coeff, 0.05)
        self.emb = nn.Embedding(vocab, d_model)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.normal_(self.pos, std=0.02)
        self.proj = nn.Linear(d_model, d_model)        # SSL projection head
        if encoder == "vjepa":
            self.predictor = nn.Linear(d_model, d_model)
        self.mask_token = nn.Parameter(torch.zeros(d_model)) if encoder == "vjepa" else None

    def forward(self, x):
        B, L = x.shape
        e = self.emb(x) + self.pos[:, :L]
        return e

    def emb_at(self, x_t, t: int):
        """단일 토큰 임베딩 + 위치 t 인코딩 — 스트리밍 step용. x_t:[B] → [B,d]."""
        return self.emb(x_t) + self.pos[:, t]

    def aux_loss(self, e):
        """encoder 변형별 SSL aux loss(표현 정규화). invariance_coeff로 가중."""
        if self.coeff <= 0:
            return e.new_zeros(())
        z = self.proj(e)
        if self.encoder == "vicreg":
            # variance + covariance 정규화(VICReg 핵심항)
            zc = z - z.mean(dim=(0, 1), keepdim=True)
            std = torch.sqrt(zc.var(dim=(0, 1)) + 1e-4)
            var_loss = F.relu(1.0 - std).mean()
            zf = zc.reshape(-1, zc.shape[-1])
            cov = (zf.T @ zf) / max(1, zf.shape[0] - 1)
            off = cov - torch.diag(torch.diag(cov))
            cov_loss = off.pow(2).sum() / z.shape[-1]
            return self.coeff * (var_loss + cov_loss)
        if self.encoder == "vjepa":
            # 일부 위치를 마스킹하고 컨텍스트로 임베딩 예측(JEPA식)
            B, L, D = z.shape
            m = (torch.rand(B, L, device=z.device) < 0.3)
            ctx = e.clone()
            ctx[m] = self.mask_token
            pred = self.predictor(self.proj(ctx))
            tgt = z.detach()
            if m.any():
                return self.coeff * F.mse_loss(pred[m], tgt[m])
            return z.new_zeros(())
        if self.encoder == "dinov2":
            # projected feature whitening(centering + entropy 유사 정규화)
            zc = z - z.mean(dim=(0, 1), keepdim=True)
            p = F.softmax(zc, dim=-1)
            entropy = -(p * (p + 1e-9).log()).sum(-1).mean()
            return self.coeff * (-entropy * 0.1 + zc.pow(2).mean() * 0.01)
        return e.new_zeros(())


# ── 기억 믹서 ────────────────────────────────────────────────────────────────
class MemoryMixer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, base_rule: str,
                 aggregation: str, segmentation: str, segment_len: int,
                 top_k: int = 4, n_persist: int = 8):
        super().__init__()
        assert d_model % n_heads == 0
        self.d = d_model
        self.h = n_heads
        self.dh = d_model // n_heads
        self.base_rule = base_rule
        self.aggregation = aggregation
        self.window = max(1, segment_len)
        self.top_k = min(top_k, n_heads)
        self.decay_floor = 0.9 if segmentation == "logarithmic" else 0.0

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        if base_rule in ("dla", "titans"):
            self.decay = nn.Linear(d_model, n_heads)       # head별 스칼라 감쇠
            nn.init.constant_(self.decay.bias, 4.0)        # 초기 a≈1(망각 최소)에서 학습 시작
        if base_rule == "titans":
            self.pk = nn.Parameter(torch.randn(n_heads, n_persist, self.dh) * 0.02)
            self.pv = nn.Parameter(torch.randn(n_heads, n_persist, self.dh) * 0.02)
        # aggregation별 결합 가중
        if aggregation in ("grm", "ssc"):
            self.head_score = nn.Linear(d_model, n_heads)
        if aggregation == "soup":
            self.o = nn.Linear(self.dh, d_model, bias=False)   # head별 d_model 투영 후 평균
        else:
            self.o = nn.Linear(d_model, d_model, bias=False)
        self.hnorm = RMSNorm(self.dh)

    def _heads(self, t):
        B, L, _ = t.shape
        return t.view(B, L, self.h, self.dh).transpose(1, 2)  # [B,H,L,dh]

    def forward(self, x):
        B, L, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = self._heads(q), self._heads(k), self._heads(v)  # [B,H,L,dh]
        scale = 1.0 / math.sqrt(self.dh)

        if self.base_rule == "swla":
            scores = (q @ k.transpose(-1, -2)) * scale          # [B,H,L,L]
            idx = torch.arange(L, device=x.device)
            causal = idx[None, :] <= idx[:, None]
            window = idx[None, :] > (idx[:, None] - self.window)
            mask = (causal & window)[None, None]
            scores = scores.masked_fill(~mask, float("-inf"))
            attn = scores.softmax(dim=-1)
            o = attn @ v                                         # [B,H,L,dh]
        else:
            # retention식 스칼라 감쇠 행렬
            if self.base_rule in ("dla", "titans"):
                a = torch.sigmoid(self.decay(x)).transpose(1, 2)  # [B,H,L] in (0,1)
                a = self.decay_floor + (1.0 - self.decay_floor) * a
            else:  # linear
                a = torch.ones(B, self.h, L, device=x.device, dtype=x.dtype)
            logA = torch.cumsum(torch.log(a + 1e-9), dim=-1)      # [B,H,L]
            D = logA[..., :, None] - logA[..., None, :]           # [B,H,L,L] = logA_t - logA_s
            idx = torch.arange(L, device=x.device)
            causal = (idx[None, :] <= idx[:, None])[None, None]   # s<=t
            D = torch.where(causal, torch.exp(D), torch.zeros_like(D))
            # 표준 선형 어텐션: feature map(elu+1)로 비음수화 후 분모 정규화 → 선택적·안정적
            qf = F.elu(q) + 1.0
            kf = F.elu(k) + 1.0
            scores = (qf @ kf.transpose(-1, -2)) * D              # [B,H,L,L] (>=0)
            denom = scores.sum(-1, keepdim=True).clamp_min(1e-6)
            o = (scores @ v) / denom                              # [B,H,L,dh]
            if self.base_rule == "titans":
                # 영속 메모리 읽기(softmax attention to learnable kv) — 장기기억
                pk = self.pk[None].expand(B, -1, -1, -1)          # [B,H,P,dh]
                pv = self.pv[None].expand(B, -1, -1, -1)
                ps = (q @ pk.transpose(-1, -2)) * scale           # [B,H,L,P]
                o = o + ps.softmax(dim=-1) @ pv

        o = self.hnorm(o)                                         # head별 정규화
        return self._aggregate(o, x)

    def forward_chunked(self, x, chunk: int = 128):
        """O(L) 청크 병렬 선형어텐션(표준 GLA) — L×L 행렬 없이 청크별 C×C + 상태 carry.

        naive forward(O(L²))와 수학적 동치. 청크 내부는 작은 C×C로 계산하고, 청크 간에는
        running state S,z를 이어받는다 → 메모리 O(L·C + dh²). swla는 해당 없음(forward 위임).
        """
        if not self.is_rnn:
            return self.forward(x)
        B, L, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = self._heads(q), self._heads(k), self._heads(v)   # [B,H,L,dh]
        qf, kf = F.elu(q) + 1.0, F.elu(k) + 1.0
        if self.base_rule in ("dla", "titans"):
            a = torch.sigmoid(self.decay(x)).transpose(1, 2)
            a = self.decay_floor + (1.0 - self.decay_floor) * a    # [B,H,L]
        else:
            a = torch.ones(B, self.h, L, device=x.device, dtype=x.dtype)

        S = torch.zeros(B, self.h, self.dh, self.dh, device=x.device, dtype=x.dtype)
        z = torch.zeros(B, self.h, self.dh, device=x.device, dtype=x.dtype)
        out = torch.empty(B, self.h, L, self.dh, device=x.device, dtype=x.dtype)
        for c0 in range(0, L, chunk):
            c1 = min(c0 + chunk, L)
            qc, kc, vc = qf[:, :, c0:c1], kf[:, :, c0:c1], v[:, :, c0:c1]  # [B,H,C,dh]
            ac = a[:, :, c0:c1]                                            # [B,H,C]
            g = torch.cumprod(ac, dim=-1)                                  # g_i=∏_{0..i} a  [B,H,C]
            C = c1 - c0
            ratio = g[..., :, None] / g[..., None, :]                      # g_i/g_j [B,H,C,C]
            idx = torch.arange(C, device=x.device)
            causal = (idx[None, :] <= idx[:, None])[None, None]
            ratio = torch.where(causal, ratio, torch.zeros_like(ratio))
            sc_intra = (qc @ kc.transpose(-1, -2)) * ratio                 # [B,H,C,C]
            o_intra = sc_intra @ vc                                        # [B,H,C,dh]
            den_intra = sc_intra.sum(-1)                                   # [B,H,C]
            # inter: carry S에서의 기여 (g_i 만큼 감쇠)
            o_inter = g[..., :, None] * (qc @ S)                           # [B,H,C,dh]
            den_inter = g * (qc * z[:, :, None, :]).sum(-1)                # [B,H,C]
            den = (den_intra + den_inter).clamp_min(1e-6)[..., None]
            out[:, :, c0:c1] = (o_intra + o_inter) / den
            # carry 갱신(청크 끝까지 감쇠)
            gC = g[..., -1]                                                # [B,H]
            re = (gC[..., None] / g)                                       # gC/g_j [B,H,C]
            kf_s = kc * re[..., None]                                      # [B,H,C,dh]
            S = gC[..., None, None] * S + torch.einsum("bhcd,bhce->bhde", kf_s, vc)
            z = gC[..., None] * z + kf_s.sum(2)

        if self.base_rule == "titans":
            scale = 1.0 / math.sqrt(self.dh)
            pk = self.pk[None].expand(B, -1, -1, -1)
            pv = self.pv[None].expand(B, -1, -1, -1)
            ps = (q @ pk.transpose(-1, -2)) * scale
            out = out + ps.softmax(dim=-1) @ pv

        out = self.hnorm(out)
        return self._aggregate(out, x)

    @property
    def is_rnn(self) -> bool:
        """O(1) 상태 재귀(RNN)로 추론 가능한 규칙? swla는 윈도우 bounded 캐시(예외)."""
        return self.base_rule in ("linear", "dla", "titans")

    def state_bytes(self, batch: int) -> int:
        """추론 시 캐싱하는 고정 크기 상태(S[dh,dh] + z[dh]) 바이트 — 시퀀스 길이 무관."""
        per = self.h * (self.dh * self.dh + self.dh)
        return batch * per * 4  # float32

    def forward_recurrent(self, x):
        """재귀(RNN) 추론 — 고정 크기 상태 S,z를 토큰마다 갱신. 병렬형과 수학적 동치.

        선형 어텐션 이중성:
          S_t = a_t·S_{t-1} + φ(k_t)⊗v_t ,  z_t = a_t·z_{t-1} + φ(k_t)
          o_t = (φ(q_t)·S_t) / (φ(q_t)·z_t)
        → L×L 어텐션 행렬을 만들지 않음(메모리 캐싱 최적화). 상태는 O(dh²)로 길이 무관.
        """
        if not self.is_rnn:
            # swla: 윈도우 캐시(O(W)) — RNN 상태는 아니지만 길이무관 bounded. 병렬형 재사용.
            return self.forward(x)

        B, L, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = self._heads(q), self._heads(k), self._heads(v)  # [B,H,L,dh]
        qf = F.elu(q) + 1.0
        kf = F.elu(k) + 1.0
        if self.base_rule in ("dla", "titans"):
            a = torch.sigmoid(self.decay(x)).transpose(1, 2)      # [B,H,L]
            a = self.decay_floor + (1.0 - self.decay_floor) * a
        else:
            a = torch.ones(B, self.h, L, device=x.device, dtype=x.dtype)

        S = torch.zeros(B, self.h, self.dh, self.dh, device=x.device, dtype=x.dtype)
        z = torch.zeros(B, self.h, self.dh, device=x.device, dtype=x.dtype)
        o = torch.empty(B, self.h, L, self.dh, device=x.device, dtype=x.dtype)
        for t in range(L):                                        # 토큰 단위 재귀(상태 캐싱)
            at = a[:, :, t]                                       # [B,H]
            kt, vt, qt = kf[:, :, t], v[:, :, t], qf[:, :, t]     # [B,H,dh]
            S = at[..., None, None] * S + kt[..., :, None] * vt[..., None, :]
            z = at[..., None] * z + kt
            num = torch.einsum("bhd,bhde->bhe", qt, S)           # φ(q)·S
            den = (qt * z).sum(-1, keepdim=True).clamp_min(1e-6)
            o[:, :, t] = num / den

        if self.base_rule == "titans":                           # 영속 메모리(정적, 상태 불필요)
            scale = 1.0 / math.sqrt(self.dh)
            pk = self.pk[None].expand(B, -1, -1, -1)
            pv = self.pv[None].expand(B, -1, -1, -1)
            ps = (q @ pk.transpose(-1, -2)) * scale
            o = o + ps.softmax(dim=-1) @ pv

        o = self.hnorm(o)
        return self._aggregate(o, x)

    # ── 진정한 토큰 스트리밍(step 단위, O(1)/스텝) ───────────────────────────
    def init_state(self, batch: int, device, dtype) -> dict:
        """스트리밍 상태 초기화. rnn: 고정 S,z. swla: 윈도우 링버퍼(O(W))."""
        if self.is_rnn:
            return {"S": torch.zeros(batch, self.h, self.dh, self.dh, device=device, dtype=dtype),
                    "z": torch.zeros(batch, self.h, self.dh, device=device, dtype=dtype)}
        return {"kbuf": None, "vbuf": None}  # swla: 마지막 W개 (k,v)

    def step(self, x_t, state: dict):
        """토큰 1개 처리. x_t:[B,d] → o:[B,d], state 갱신(인플레이스 dict 반환).

        rnn은 고정 상태만 갱신(시퀀스 길이 무관 O(1)). swla는 마지막 W개 (k,v) 윈도우 유지.
        """
        B = x_t.shape[0]
        q, k, v = self.qkv(x_t).chunk(3, dim=-1)                  # [B,d] each
        q = q.view(B, self.h, self.dh)
        k = k.view(B, self.h, self.dh)
        v = v.view(B, self.h, self.dh)
        scale = 1.0 / math.sqrt(self.dh)

        if self.is_rnn:
            qf = F.elu(q) + 1.0
            kf = F.elu(k) + 1.0
            if self.base_rule in ("dla", "titans"):
                a = torch.sigmoid(self.decay(x_t))               # [B,H]
                a = self.decay_floor + (1.0 - self.decay_floor) * a
            else:
                a = torch.ones(B, self.h, device=x_t.device, dtype=x_t.dtype)
            S = a[..., None, None] * state["S"] + kf[..., :, None] * v[..., None, :]
            z = a[..., None] * state["z"] + kf
            num = torch.einsum("bhd,bhde->bhe", qf, S)           # [B,H,dh]
            den = (qf * z).sum(-1, keepdim=True).clamp_min(1e-6)
            o = num / den
            state["S"], state["z"] = S, z
            if self.base_rule == "titans":
                pk = self.pk[None].expand(B, -1, -1, -1)          # [B,H,P,dh]
                pv = self.pv[None].expand(B, -1, -1, -1)
                ps = torch.einsum("bhd,bhpd->bhp", q, pk) * scale # [B,H,P]
                o = o + torch.einsum("bhp,bhpd->bhd", ps.softmax(-1), pv)
        else:
            # swla: 윈도우 링버퍼에 현재 (k,v) 추가 후 마지막 W개에 softmax attention
            kt, vt = k[:, :, None, :], v[:, :, None, :]           # [B,H,1,dh]
            kbuf = kt if state["kbuf"] is None else torch.cat([state["kbuf"], kt], dim=2)
            vbuf = vt if state["vbuf"] is None else torch.cat([state["vbuf"], vt], dim=2)
            kbuf, vbuf = kbuf[:, :, -self.window:], vbuf[:, :, -self.window:]
            state["kbuf"], state["vbuf"] = kbuf, vbuf
            scores = torch.einsum("bhd,bhwd->bhw", q, kbuf) * scale  # [B,H,W']
            attn = scores.softmax(-1)
            o = torch.einsum("bhw,bhwd->bhd", attn, vbuf)         # [B,H,dh]

        o = self.hnorm(o)                                         # [B,H,dh]
        out = self._aggregate(o[:, :, None, :], x_t[:, None, :])  # [B,1,d]
        return out[:, 0], state

    def _aggregate(self, o, x):
        # o: [B,H,L,dh]
        B, H, L, dh = o.shape
        oh = o.transpose(1, 2)                                    # [B,L,H,dh]
        if self.aggregation == "soup":
            # 각 head를 d_model로 투영 후 평균(model-soup식, dh 병목 없음)
            return self.o(oh).mean(dim=2)                         # [B,L,H,d]→[B,L,d]
        if self.aggregation == "grm":
            g = torch.sigmoid(self.head_score(x))                # [B,L,H]
            oh = oh * g[..., None]
        elif self.aggregation == "ssc":
            # 상위 top_k head만 sigmoid 게이트로 활성(스케일 보존), 나머지 0
            s = self.head_score(x)                                # [B,L,H]
            _, topi = s.topk(self.top_k, dim=-1)
            keep = torch.zeros_like(s).scatter_(-1, topi, 1.0)
            gate = torch.sigmoid(s) * keep                        # 비선택 head=0
            oh = oh * gate[..., None]
        # residual / grm / ssc: concat heads → proj
        return self.o(oh.reshape(B, L, H * dh))


class Block(nn.Module):
    def __init__(self, cfg, segment_len, top_k):
        super().__init__()
        d = cfg["d_model"]
        self.n1 = RMSNorm(d)
        self.mix = MemoryMixer(d, cfg["n_heads"], cfg["base_rule"],
                               cfg["aggregation"], cfg["segmentation"],
                               segment_len, top_k)
        self.n2 = RMSNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        x = x + self.mix(self.n1(x))
        x = x + self.mlp(self.n2(x))
        return x

    def forward_recurrent(self, x):
        x = x + self.mix.forward_recurrent(self.n1(x))            # 믹서만 재귀, 나머지는 동일
        x = x + self.mlp(self.n2(x))
        return x

    def forward_chunked(self, x, chunk: int = 128):
        x = x + self.mix.forward_chunked(self.n1(x), chunk)
        x = x + self.mlp(self.n2(x))
        return x

    def step(self, x_t, state: dict):
        o, state = self.mix.step(self.n1(x_t), state)
        x_t = x_t + o
        x_t = x_t + self.mlp(self.n2(x_t))
        return x_t, state


class GrowingMemoryModel(nn.Module):
    def __init__(self, cfg: dict, vocab: int, max_len: int):
        super().__init__()
        d = cfg["d_model"]
        ssl = cfg.get("ssl") or {"encoder": "none", "invariance_coeff": "low"}
        self.front = SSLFront(vocab, d, max_len, ssl.get("encoder", "none"),
                              ssl.get("invariance_coeff", "low"))
        self.init_mode = cfg.get("init_mode", "independent")
        if self.init_mode == "checkpoint":
            self.mem_init = nn.Linear(d, d)                       # 컨텍스트 평균 → 메모리 초기화
        seg = cfg.get("segment_len", 128)
        top_k = cfg.get("top_k", 4)
        self.blocks = nn.ModuleList([Block(cfg, seg, top_k) for _ in range(cfg["n_layers"])])
        self.norm = RMSNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.front.emb.weight                 # weight tying
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear) and m.weight is not None:
            nn.init.normal_(m.weight, std=0.02)
        if isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, x, return_aux: bool = False):
        e = self.front(x)
        if self.init_mode == "checkpoint":
            e = e + self.mem_init(e.mean(dim=1, keepdim=True))    # 컨텍스트 요약 주입
        h = e
        for blk in self.blocks:
            h = blk(h)
        logits = self.head(self.norm(h))
        if return_aux:
            return logits, self.front.aux_loss(e)
        return logits

    def forward_chunked(self, x, chunk: int = 128):
        """O(L) 청크 병렬 — 학습/prefill용. L×L 없이 청크별 계산(naive forward와 동치)."""
        e = self.front(x)
        if self.init_mode == "checkpoint":
            e = e + self.mem_init(e.mean(dim=1, keepdim=True))
        h = e
        for blk in self.blocks:
            h = blk.forward_chunked(h, chunk)
        return self.head(self.norm(h))

    @torch.no_grad()
    def forward_recurrent(self, x):
        """재귀(RNN) 추론 — L×L 어텐션 없이 고정 상태 캐싱으로 처리(엣지 고정상태 추론, §8).

        병렬형 forward()와 동치(동일 입력 → 동일 출력, fp 오차 내). 메모리는 길이 무관 상수 상태.
        """
        e = self.front(x)
        if self.init_mode == "checkpoint":
            e = e + self.mem_init(e.mean(dim=1, keepdim=True))    # 컨텍스트 요약(고정길이 입력 가정)
        h = e
        for blk in self.blocks:
            h = blk.forward_recurrent(h)
        return self.head(self.norm(h))

    def state_bytes(self, batch: int) -> int:
        """전 레이어 재귀 상태 합(바이트) — 시퀀스 길이와 무관한 고정 캐시 크기."""
        return sum(blk.mix.state_bytes(batch) for blk in self.blocks
                   if blk.mix.is_rnn)

    # ── 진정한 토큰 스트리밍 API(엣지: 토큰 1개씩, 고정 상태) ─────────────────
    def init_states(self, batch: int, device, dtype) -> list:
        return [blk.mix.init_state(batch, device, dtype) for blk in self.blocks]

    @torch.no_grad()
    def step(self, token, t: int, states: list):
        """토큰 1개(+위치 t) → 다음 토큰 logits. states를 갱신해 반환.

        init_mode=checkpoint는 전체 컨텍스트 평균이 필요해 스트리밍 불가 → independent만 지원.
        """
        if self.init_mode == "checkpoint":
            raise RuntimeError("streaming step은 init_mode=independent에서만 지원됩니다")
        h = self.front.emb_at(token, t)                          # [B,d]
        for blk, st in zip(self.blocks, states):
            h, _ = blk.step(h, st)
        logits = self.head(self.norm(h))                        # [B,V]
        return logits, states

    @torch.no_grad()
    def stream(self, x):
        """시퀀스를 토큰 단위로 스트리밍 처리 → [B,L,V]. forward()와 동치(independent)."""
        B, L = x.shape
        states = self.init_states(B, x.device, self.head.weight.dtype)
        outs = []
        for t in range(L):
            logits, states = self.step(x[:, t], t, states)
            outs.append(logits)
        return torch.stack(outs, dim=1)


def build_real(cfg: dict, vocab: int, max_len: int) -> GrowingMemoryModel:
    return GrowingMemoryModel(cfg, vocab, max_len)
