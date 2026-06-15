#!/usr/bin/env python3
"""효율 재검증 — 검증된 알고리즘(fla DeltaNet) vs 표준 트랜스포머(FlashAttention).

이전 compare3는 from-scratch 선형이었음. 여기선 정확·검증된 deltanet으로 메모리·시간을 다시 측정.
같은 차원/깊이, bf16, batch1, 길이별 forward peak 메모리 + 시간. fla는 O(L) mem·O(L) time,
트랜스포머(flash)는 O(L) mem·O(L²) time 예상.

    ~/gm_venv/bin/python efficiency_bench.py
"""
from __future__ import annotations
import sys, os, time, torch
import torch.nn as nn
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
sys.path.insert(0, os.path.join(_here, "..", "..", "tutorial", "autoresearch"))  # compare3
from compare3 import Transformer, D, H, LAYERS, VOCAB
from fla.layers import DeltaNet

DEV="cuda"; DT=torch.bfloat16

class DeltaNetLM(nn.Module):
    def __init__(s, vocab, d, h, layers, max_len):
        super().__init__()
        s.emb=nn.Embedding(vocab,d); s.pos=nn.Parameter(torch.zeros(1,max_len,d))
        s.bl=nn.ModuleList([nn.ModuleDict({"n1":nn.LayerNorm(d),"mix":DeltaNet(hidden_size=d,num_heads=h),
            "n2":nn.LayerNorm(d),"mlp":nn.Sequential(nn.Linear(d,4*d),nn.GELU(),nn.Linear(4*d,d))}) for _ in range(layers)])
        s.norm=nn.LayerNorm(d); s.head=nn.Linear(d,vocab,bias=False)
    def forward(s,idx):
        h=s.emb(idx)+s.pos[:,:idx.shape[1]]
        for b in s.bl:
            o=b["mix"](b["n1"](h)); o=o[0] if isinstance(o,tuple) else o
            h=h+o; h=h+b["mlp"](b["n2"](h))
        return s.head(s.norm(h))

def peak(): torch.cuda.synchronize(); return torch.cuda.max_memory_allocated()/1e9
def run(fn, iters=3):
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        fn(); torch.cuda.synchronize(); t0=time.time()
        for _ in range(iters): fn()
        torch.cuda.synchronize()
    return round((time.time()-t0)/iters*1000,1), round(peak(),2)

torch.manual_seed(0)
maxL=140000
dn=DeltaNetLM(VOCAB,D,H,LAYERS,maxL).to(DEV).to(DT).eval()
tf=Transformer(VOCAB,D,H,LAYERS,maxL).to(DEV).to(DT).eval()
print(f"d={D} h={H} L={LAYERS} bf16 batch1")
print(f"{'L':>7} {'DeltaNet ms':>12} {'DeltaNet GB':>12} {'TF ms':>9} {'TF GB':>8} {'TF/DN time':>11}")
for Ln in [2048, 8192, 32768, 65536, 131072]:
    x=torch.randint(1,VOCAB,(1,Ln),device=DEV)
    try: dn_t,dn_m=run(lambda: dn(x))
    except RuntimeError: dn_t,dn_m="OOM","-"; torch.cuda.empty_cache()
    try: tf_t,tf_m=run(lambda: tf(x))
    except RuntimeError: tf_t,tf_m="OOM","-"; torch.cuda.empty_cache()
    ratio=round(tf_t/dn_t,2) if (isinstance(tf_t,float) and isinstance(dn_t,float)) else "-"
    print(f"{Ln:>7} {str(dn_t):>12} {str(dn_m):>12} {str(tf_t):>9} {str(tf_m):>8} {str(ratio):>11}",flush=True)
