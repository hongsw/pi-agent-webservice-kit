#!/usr/bin/env python3
"""보고서용 차트 생성 — 세션 실측 수치를 PNG로(report/figures/). matplotlib만 사용."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), "..", "report", "figures")
os.makedirs(OUT, exist_ok=True)
def save(fig, name):
    fig.tight_layout(); p = os.path.join(OUT, name); fig.savefig(p, dpi=130); plt.close(fig); print("wrote", p)

# 1) 연산 시간 vs L (DeltaNet O(L) vs FlashAttn O(L²))  — efficiency_bench.py 실측
L = [2048, 8192, 32768, 65536, 131072]
dn = [4.0, 3.8, 6.9, 14.8, 31.2]; tf = [0.8, 2.1, 18.5, 65.4, 244.0]
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(L, tf, "o-", color="#d9534f", label="Transformer (FlashAttn) O(L²)")
ax.plot(L, dn, "o-", color="#3fb950", label="DeltaNet (linear) O(L)")
ax.set_xscale("log", base=2); ax.set_yscale("log")
ax.set_xlabel("sequence length L"); ax.set_ylabel("forward time (ms)")
ax.set_title("연산 시간 vs 길이 — 128K에서 DeltaNet 7.8× 빠름"); ax.legend(); ax.grid(True, which="both", alpha=.3)
save(fig, "fig1_compute_time.png")

# 2) 추론 메모리: 트랜스포머 KV캐시 O(L) vs 재귀 상태 O(1)  — infer_bench.py 실측
L2 = [8192, 131072, 1048576, 2097152]; kv = [0.08, 1.17, 9.69, 19.35]; state = [0.0259]*4
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(L2, kv, "o-", color="#d9534f", label="Transformer KV cache  O(L)")
ax.plot(L2, state, "o-", color="#4f8cff", label="Recurrent state  O(1) (132KB)")
ax.set_xscale("log", base=2); ax.set_yscale("log")
ax.axhline(24, ls="--", color="#888", lw=1); ax.text(9000, 26, "RTX 4090 24GB", color="#888", fontsize=8)
ax.set_xlabel("context length L"); ax.set_ylabel("inference memory (GB)")
ax.set_title("추론 메모리 — KV캐시 폭증 vs 상수 상태 (~992×@16K)"); ax.legend(); ax.grid(True, which="both", alpha=.3)
save(fig, "fig2_inference_memory.png")

# 3) vetted 구현 recall (fla_validate.py)
names = ["deltanet", "gated_deltanet", "retention", "titans", "linear", "gla"]
rec = [1.0, 1.0, 1.0, 1.0, 0.54, 0.06]
fig, ax = plt.subplots(figsize=(6.5, 4))
cols = ["#3fb950" if r >= .9 else ("#d29922" if r > .2 else "#d9534f") for r in rec]
ax.bar(names, rec, color=cols)
ax.axhline(0.062, ls="--", color="#888", lw=1); ax.text(0, 0.09, "chance", color="#888", fontsize=8)
ax.set_ylabel("MQAR recall"); ax.set_ylim(0, 1.05)
ax.set_title("검증된 구현 회상 — deltanet/retention/titans = 1.0")
plt.setp(ax.get_xticklabels(), rotation=20, ha="right"); ax.grid(True, axis="y", alpha=.3)
save(fig, "fig3_recall_by_variant.png")

# 4) grokking 곡선 (titans seq=512, 25k)
gs = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]
gr = [.324,.332,.335,.346,.335,.339,.341,.342,.336,.336,.337,.337,.326,.337,.604,.789,.939,.986,.995,.997,1,1,1,1,1]
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.plot([g*1000 for g in gs], gr, "-", color="#4f8cff")
ax.axvline(15000, ls="--", color="#3fb950", lw=1); ax.text(15500, .2, "grok@15k", color="#3fb950", fontsize=9)
ax.set_xlabel("training steps"); ax.set_ylabel("recall"); ax.set_ylim(0, 1.05)
ax.set_title("grokking — titans 긴 컨텍스트(seq=512) 0.33 정체→1.0 급점프"); ax.grid(True, alpha=.3)
save(fig, "fig4_grokking.png")
print("done")
