#!/usr/bin/env python3
"""vetted growing-memory 구현 축별 검증 — 단일 고정 LayerNorm 스캐폴드에서 MQAR recall.

정확 구현 먼저: flash-linear-attention(fla) + titans-pytorch(lucidrains)의 검증된 모듈을
같은 스캐폴드에 끼워 같은 task로 비교. (from-scratch 근사 대체)

    ~/gm_venv/bin/python fla_validate.py
"""
from __future__ import annotations
import sys, os, time, torch
import torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from growing_memory.data import make_mqar_batch

DEV="cuda" if torch.cuda.is_available() else "cpu"
VOCAB, DIM, HEADS, DEPTH, L, PAIRS, B, STEPS = 32, 128, 4, 2, 64, 2, 64, 4000

def wrap(mod):
    """fla/titans 레이어를 [B,L,d]->[B,L,d]로 정규화(tuple[0])."""
    class W(nn.Module):
        def __init__(s): super().__init__(); s.m=mod
        def forward(s,x):
            o=s.m(x); return o[0] if isinstance(o,tuple) else o
    return W()

def factories():
    from fla.layers import (LinearAttention, GatedLinearAttention, DeltaNet,
                            GatedDeltaNet, MultiScaleRetention)
    f={
      "linear(fla)":      lambda: wrap(LinearAttention(hidden_size=DIM, num_heads=HEADS)),
      "gla/dla(fla)":     lambda: wrap(GatedLinearAttention(hidden_size=DIM, num_heads=HEADS)),
      "deltanet(fla)":    lambda: wrap(DeltaNet(hidden_size=DIM, num_heads=HEADS)),
      "gated_deltanet(fla)": lambda: wrap(GatedDeltaNet(hidden_size=DIM, num_heads=HEADS)),
      "retention(fla)":   lambda: wrap(MultiScaleRetention(hidden_size=DIM, num_heads=HEADS)),
    }
    try:
        from titans_pytorch import NeuralMemory
        f["titans(lucidrains)"]=lambda: wrap(NeuralMemory(dim=DIM, chunk_size=32))
    except Exception as e:
        print("titans skip:", str(e)[:60])
    return f

class LM(nn.Module):
    def __init__(s, fac):
        super().__init__()
        s.emb=nn.Embedding(VOCAB,DIM); s.pos=nn.Parameter(torch.zeros(1,L,DIM)); nn.init.normal_(s.pos,std=0.02)
        s.bl=nn.ModuleList([nn.ModuleDict({"n1":nn.LayerNorm(DIM),"mix":fac(),
            "n2":nn.LayerNorm(DIM),"mlp":nn.Sequential(nn.Linear(DIM,4*DIM),nn.GELU(),nn.Linear(4*DIM,DIM))}) for _ in range(DEPTH)])
        s.norm=nn.LayerNorm(DIM); s.head=nn.Linear(DIM,VOCAB,bias=False)
    def forward(s,idx):
        h=s.emb(idx)+s.pos[:,:idx.shape[1]]
        for b in s.bl: h=h+b["mix"](b["n1"](h)); h=h+b["mlp"](b["n2"](h))
        return s.head(s.norm(h))

def run(name, fac):
    torch.manual_seed(0); m=LM(fac).to(DEV)
    opt=torch.optim.AdamW(m.parameters(),lr=3e-3,weight_decay=0.1)
    tg=torch.Generator().manual_seed(0); eg=torch.Generator().manual_seed(99)
    def ev():
        m.eval(); c=t=0
        with torch.no_grad():
            eg.manual_seed(99)
            for _ in range(6):
                i,tt,qm=make_mqar_batch(B,L,PAIRS,VOCAB,DEV,eg); p=m(i).argmax(-1); c+=(p[qm]==tt[qm]).sum().item(); t+=qm.sum().item()
        return c/t if t else 0
    cur=[]; t0=time.time()
    for s in range(STEPS):
        m.train(); i,tt,_=make_mqar_batch(B,L,PAIRS,VOCAB,DEV,tg)
        loss=F.cross_entropy(m(i).reshape(-1,VOCAB),tt.reshape(-1),ignore_index=-100)
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
        if (s+1)%1000==0: cur.append((s+1,round(ev(),3)))
    return round(ev(),3), cur, round(time.time()-t0,0)

print(f"MQAR vocab{VOCAB} pairs{PAIRS} L{L} dim{DIM} chance≈{1/(VOCAB//2):.3f} steps{STEPS} dev={DEV}")
for name,fac in factories().items():
    try:
        r,cur,sec=run(name,fac); print(f"{name:>22}: recall={r}  {cur}  ({sec}s)",flush=True)
    except Exception as e:
        print(f"{name:>22}: ERR {str(e)[:90]}",flush=True)
