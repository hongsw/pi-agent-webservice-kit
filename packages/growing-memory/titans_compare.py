#!/usr/bin/env python3
"""titans 정확 구현 교차검증 — 우리 NeuralMemoryExact vs lucidrains titans-pytorch NeuralMemory.

동일한 미니 LM 스캐폴드(emb+pos → [norm→mem→res → norm→mlp→res]×depth → head)에 메모리 모듈만
교체하고, 같은 MQAR(연관회상)로 masked CE 학습 → recall 곡선 비교. + 기존 근사 titans도 비교.

    ~/gm_venv/bin/python titans_compare.py        # (torch + titans-pytorch 필요)
"""
from __future__ import annotations
import sys, os, time, torch
import torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from growing_memory.titans_exact import NeuralMemoryExact
from growing_memory.data import make_mqar_batch

DEV="cuda" if torch.cuda.is_available() else "cpu"
VOCAB, DIM, DEPTH, L, PAIRS, B = 32, 128, 2, 64, 2, 64

class LM(nn.Module):
    def __init__(self, mem_factory):
        super().__init__()
        self.emb=nn.Embedding(VOCAB,DIM); self.pos=nn.Parameter(torch.zeros(1,L,DIM)); nn.init.normal_(self.pos,std=0.02)
        self.blocks=nn.ModuleList()
        for _ in range(DEPTH):
            self.blocks.append(nn.ModuleDict({"n1":nn.LayerNorm(DIM),"mem":mem_factory(),
                "n2":nn.LayerNorm(DIM),"mlp":nn.Sequential(nn.Linear(DIM,4*DIM),nn.GELU(),nn.Linear(4*DIM,DIM))}))
        self.norm=nn.LayerNorm(DIM); self.head=nn.Linear(DIM,VOCAB,bias=False)
    def forward(self, idx):
        h=self.emb(idx)+self.pos[:,:idx.shape[1]]
        for b in self.blocks:
            o=b["mem"](b["n1"](h)); o=o[0] if isinstance(o,tuple) else o
            h=h+o; h=h+b["mlp"](b["n2"](h))
        return self.head(self.norm(h))

def lucidrains_mem():
    from titans_pytorch import NeuralMemory
    return NeuralMemory(dim=DIM, chunk_size=32)

def train_eval(name, mem_factory, steps=3000):
    torch.manual_seed(0); m=LM(mem_factory).to(DEV)
    opt=torch.optim.AdamW(m.parameters(),lr=3e-3,weight_decay=0.1)
    tgen=torch.Generator().manual_seed(0); egen=torch.Generator().manual_seed(99)
    def ev():
        m.eval(); c=t=0
        with torch.no_grad():
            egen.manual_seed(99)
            for _ in range(6):
                inp,tgt,qm=make_mqar_batch(B,L,PAIRS,VOCAB,DEV,egen)
                p=m(inp).argmax(-1); c+=(p[qm]==tgt[qm]).sum().item(); t+=qm.sum().item()
        return c/t if t else 0
    curve=[]; t0=time.time()
    for s in range(steps):
        m.train(); inp,tgt,_=make_mqar_batch(B,L,PAIRS,VOCAB,DEV,tgen)
        loss=F.cross_entropy(m(inp).reshape(-1,VOCAB),tgt.reshape(-1),ignore_index=-100)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
        if (s+1)%500==0: curve.append((s+1,round(ev(),3)))
    return {"final":round(ev(),3),"curve":curve,"secs":round(time.time()-t0,1)}

print(f"MQAR vocab{VOCAB} pairs{PAIRS} L{L} chance≈{1/(VOCAB//2):.3f} dev={DEV}")
for name,fac in [("titans_exact(우리,논문식)", lambda: NeuralMemoryExact(DIM)),
                 ("lucidrains NeuralMemory", lucidrains_mem)]:
    try:
        r=train_eval(name,fac)
        print(f"{name:>26}: final_recall={r['final']}  curve={r['curve']}  ({r['secs']}s)")
    except Exception as e:
        print(f"{name:>26}: ERROR {str(e)[:80]}")
