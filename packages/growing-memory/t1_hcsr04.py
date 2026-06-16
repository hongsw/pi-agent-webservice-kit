#!/usr/bin/env python3
"""T1 — 실센서(HC-SR04) 데이터를 autoresearch 학습 루프에 연결(엣지학습노드 1단계).

실 초음파 거리 시계열(Measured)을 V 빈으로 토큰화 → 다음값 예측 자기지도 과제. 검증된
base_rule(fla deltanet 등)을 단일 스캐폴드로 실데이터 학습. predict-previous 베이스라인 대비 측정.

데이터 출처: data/hcsr04/dist_hcsr04.csv (NAS 도커가 서빙하는 '커밋 샤드'에 해당; 여기선 로컬 캐시).
    ~/gm_venv/bin/python t1_hcsr04.py
"""
from __future__ import annotations
import sys, os, csv as _csv, time, torch
import torch.nn as nn, torch.nn.functional as F
_here=os.path.dirname(os.path.abspath(__file__))
CSV=os.path.normpath(os.path.join(_here,"..","..","data","hcsr04","dist_hcsr04.csv"))
DEV="cuda" if torch.cuda.is_available() else "cpu"
V, DIM, HEADS, DEPTH, SEQ, B = 32, 128, 4, 2, 48, 64

def load_tokens(path, vocab):
    vals=[float(r["Measured"]) for r in _csv.DictReader(open(path))]
    lo,hi=min(vals),max(vals)
    toks=[min(vocab-1, int((v-lo)/(hi-lo+1e-9)*vocab)) for v in vals]
    return torch.tensor(toks, dtype=torch.long), (lo,hi)

def batch(seq, n, L, gen):
    N=seq.numel(); hi=max(1, N-L-1)
    idx=torch.randint(0, hi, (n,), generator=gen)
    win=torch.stack([seq[i:i+L+1] for i in idx])           # [n,L+1]
    return win[:,:-1].to(DEV), win[:,1:].to(DEV)

def mixer(name):
    from fla.layers import DeltaNet, GatedLinearAttention, LinearAttention, MultiScaleRetention
    cls={"deltanet":DeltaNet,"gla":GatedLinearAttention,"linear":LinearAttention,"retention":MultiScaleRetention}[name]
    m=cls(hidden_size=DIM, num_heads=HEADS)
    class W(nn.Module):
        def __init__(s): super().__init__(); s.m=m
        def forward(s,x):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o=s.m(x)
            o=o[0] if isinstance(o,tuple) else o
            return o.float()
    return W()

class LM(nn.Module):
    def __init__(s, name):
        super().__init__(); s.emb=nn.Embedding(V,DIM); s.pos=nn.Parameter(torch.zeros(1,SEQ,DIM)); nn.init.normal_(s.pos,std=0.02)
        s.bl=nn.ModuleList([nn.ModuleDict({"n1":nn.LayerNorm(DIM),"mix":mixer(name),"n2":nn.LayerNorm(DIM),
            "mlp":nn.Sequential(nn.Linear(DIM,4*DIM),nn.GELU(),nn.Linear(4*DIM,DIM))}) for _ in range(DEPTH)])
        s.norm=nn.LayerNorm(DIM); s.head=nn.Linear(DIM,V,bias=False)
    def forward(s,x):
        h=s.emb(x)+s.pos[:,:x.shape[1]]
        for b in s.bl: h=h+b["mix"](b["n1"](h)); h=h+b["mlp"](b["n2"](h))
        return s.head(s.norm(h))

def run(name, seq_tr, seq_ev, steps=2000):
    torch.manual_seed(0); m=LM(name).to(DEV)
    opt=torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=0.05)
    tg=torch.Generator().manual_seed(0); eg=torch.Generator().manual_seed(7)
    def acc():
        m.eval(); c=t=0
        with torch.no_grad():
            eg.manual_seed(7)
            for _ in range(8):
                i,tt=batch(seq_ev,B,SEQ,eg); p=m(i).argmax(-1); c+=(p==tt).sum().item(); t+=tt.numel()
        return c/t
    t0=time.time()
    for s in range(steps):
        m.train(); i,tt=batch(seq_tr,B,SEQ,tg)
        loss=F.cross_entropy(m(i).reshape(-1,V), tt.reshape(-1))
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
    return round(acc(),3), round(time.time()-t0,0)

def baseline_prev(seq_ev):
    # predict-previous: next == current 토큰일 확률(강한 베이스라인, 구간 내 상수)
    s=seq_ev; return round((s[1:]==s[:-1]).float().mean().item(),3)

if __name__=="__main__":
    seq,(lo,hi)=load_tokens(CSV, V)
    n=seq.numel()
    # 시간순 분할은 분포이동(블록 미관측) → 전 구간 윈도우로 학습/평가(시드만 분리). 퇴화 과제 진단용.
    tr=ev=seq
    print(f"T1 HC-SR04 실센서 → autoresearch  (N={n} tokens, V={V}, raw {lo:.0f}~{hi:.0f}, 전구간윈도우, dev={DEV})")
    print(f"baseline(predict-previous) next-step acc = {baseline_prev(ev)}  (chance={1/V:.3f})")
    for name in ["deltanet","retention","linear","gla"]:
        try:
            a,sec=run(name,tr,ev); print(f"  {name:>9}: next-step acc={a}  ({sec}s)",flush=True)
        except Exception as e:
            print(f"  {name:>9}: ERR {str(e)[:80]}",flush=True)
