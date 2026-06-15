#!/usr/bin/env python3
"""grokking 튜닝 — 8변형(base_rule×aggregation)이 전부 MQAR recall을 학습하도록 레시피 탐색.

grokking 가속 레버(Power et al. 2022): weight decay + 충분한 스텝 + warmup/cosine.
각 변형을 학습하며 eval recall 곡선과 'grok step'(recall>0.5 최초 도달)을 기록.

    python3 grok_tune.py --steps 6000 --wd 0.1 --lr 3e-3 --warmup 200
    python3 grok_tune.py --variants soup,titans_ssc --steps 8000 --wd 0.2   # 어려운 것만
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from autoresearch.real.model import build_real  # noqa: E402
from autoresearch.real.data import make_mqar_batch  # noqa: E402

BASE = {"segmentation": "logarithmic", "init_mode": "independent",
        "segment_len": 64, "d_model": 256, "n_layers": 4, "n_heads": 8,
        "ssl": {"encoder": "none", "invariance_coeff": "low"}}

# 8변형: base_rule 축(agg=residual) 4 + aggregation 축(base=linear) 3 + 최난조합 titans+ssc
VARIANTS = {
    "linear_residual": {"base_rule": "linear", "aggregation": "residual"},
    "swla_residual":   {"base_rule": "swla",   "aggregation": "residual"},
    "dla_residual":    {"base_rule": "dla",    "aggregation": "residual"},
    "titans_residual": {"base_rule": "titans", "aggregation": "residual"},
    "linear_grm":      {"base_rule": "linear", "aggregation": "grm"},
    "linear_soup":     {"base_rule": "linear", "aggregation": "soup"},
    "linear_ssc":      {"base_rule": "linear", "aggregation": "ssc", "top_k": 4},
    "titans_ssc":      {"base_rule": "titans", "aggregation": "ssc", "top_k": 4},
    "titans_real":     {"base_rule": "titans_real", "aggregation": "residual"},  # lucidrains 정확 Titans
}


def lr_at(step, total, base_lr, warmup):
    if step < warmup:
        return base_lr * (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return 0.5 * base_lr * (1 + math.cos(math.pi * min(1.0, p)))


def evaluate(model, rt, gen, n=6, hard=False, chunked=False):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for _ in range(n):
            pairs = rt["num_pairs"] + (2 if hard else 0)
            L = rt["seq_len"] + (rt["seq_len"] // 2 if hard else 0)
            inp, tgt, qm = make_mqar_batch(rt["batch"], L, pairs, rt["vocab"],
                                           rt["device"], gen)
            logits = model.forward_chunked(inp) if chunked else model(inp)
            if qm.any():
                pred = logits.argmax(-1)
                correct += (pred[qm] == tgt[qm]).sum().item()
                total += qm.sum().item()
    return correct / total if total else 0.0


def train_variant(name, vcfg, args, device):
    cfg = {**BASE, **vcfg}
    if args.d_model:
        cfg["d_model"] = args.d_model
    if args.n_heads:
        cfg["n_heads"] = args.n_heads          # head↓ → dh↑ → 선형 어텐션 상태(회상) 용량↑
    if args.n_layers:
        cfg["n_layers"] = args.n_layers
    rt = {"device": device, "vocab": args.vocab, "seq_len": args.seq_len,
          "num_pairs": args.pairs, "batch": args.batch}
    torch.manual_seed(0)
    model = build_real(cfg, args.vocab, max_len=args.seq_len * 2 + 4).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            betas=(0.9, 0.98), weight_decay=args.wd)
    tgen = torch.Generator().manual_seed(0)
    egen = torch.Generator().manual_seed(999)
    chance = 1.0 / (args.vocab // 2)
    grok_step = None
    curve = []
    t0 = time.time()
    for step in range(args.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, args.steps, args.lr, args.warmup)
        model.train()
        inp, tgt, _ = make_mqar_batch(args.batch, args.seq_len, args.pairs,
                                      args.vocab, device, tgen)
        if args.chunked:
            logits, aux = model.forward_chunked(inp, return_aux=True)   # O(L) 학습
        else:
            logits, aux = model(inp, return_aux=True)
        loss = F.cross_entropy(logits.reshape(-1, args.vocab),
                               tgt.reshape(-1), ignore_index=-100) + aux
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (step + 1) % args.eval_every == 0:
            egen.manual_seed(999)
            r = evaluate(model, rt, egen, chunked=args.chunked)
            curve.append((step + 1, round(r, 3)))
            if grok_step is None and r > 0.5:
                grok_step = step + 1
    egen.manual_seed(999)
    final = evaluate(model, rt, egen, chunked=args.chunked)
    egen.manual_seed(777)
    final_hard = evaluate(model, rt, egen, hard=True, chunked=args.chunked)
    return {"name": name, "final_recall": round(final, 3),
            "hard_recall": round(final_hard, 3), "grok_step": grok_step,
            "chance": round(chance, 3), "curve": curve,
            "secs": round(time.time() - t0, 1)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--eval-every", dest="eval_every", type=int, default=500)
    ap.add_argument("--vocab", type=int, default=32)
    ap.add_argument("--seq-len", dest="seq_len", type=int, default=20)
    ap.add_argument("--pairs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--d-model", dest="d_model", type=int, default=0)
    ap.add_argument("--n-heads", dest="n_heads", type=int, default=0)
    ap.add_argument("--n-layers", dest="n_layers", type=int, default=0)
    ap.add_argument("--variants", default="all")
    ap.add_argument("--chunked", action="store_true", help="O(L) chunked 학습/평가(긴 시퀀스)")
    args = ap.parse_args(argv)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    names = list(VARIANTS) if args.variants == "all" else args.variants.split(",")
    print(f"recipe: steps={args.steps} wd={args.wd} lr={args.lr} warmup={args.warmup} "
          f"task(vocab={args.vocab} pairs={args.pairs} seq={args.seq_len}) device={device}")
    chance = 1.0 / (args.vocab // 2)
    print(f"chance≈{chance:.3f}\n{'variant':<18} {'final':>6} {'hard':>6} {'grok@':>7}  curve")
    results = []
    for n in names:
        r = train_variant(n, VARIANTS[n], args, device)
        results.append(r)
        cs = " ".join(f"{s}:{v}" for s, v in r["curve"])
        print(f"{n:<18} {r['final_recall']:>6} {r['hard_recall']:>6} "
              f"{str(r['grok_step']):>7}  {cs}  ({r['secs']}s)")
    learned = sum(1 for r in results if r["final_recall"] > 5 * chance)
    print(f"\n학습 성공(>5×chance): {learned}/{len(results)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
