# growing-memory-pytorch

Unofficial, faithful PyTorch reimplementation of **Memory Caching: RNNs with Growing Memory** (Behrouz et al., [arXiv:2602.24281](https://arxiv.org/abs/2602.24281), 2026).

> The repo is named after the paper's subtitle — *RNNs with Growing Memory* — i.e. the **capability** the framework delivers. The mechanism inside is **Memory Caching (MC)**, and the scope is the paper's **full framework**: aggregation design, deep / test-time memory, segmentation, and the architecture-unification results — not the caching mechanism alone.

> **Status:** experimental / work-in-progress. Correctness is enforced by equation-level equivalence tests (see [Correctness](#correctness)); large-scale reproduction is in progress (see [Reproduction status](#reproduction-status)).

> **Disclaimer:** This is an independent, community reimplementation. It is **not** affiliated with or endorsed by the paper's authors, and no official code release is used. Where the paper underspecifies a detail, we make an explicit, documented choice (see [Assumptions](#assumptions-where-the-paper-is-underspecified)).

---

## What this implements

Memory Caching (MC) is best read not as a single trick but as a **unifying framework for growing-memory sequence models**. Its backbone is simple — split the sequence into segments, compress each into a memory state, cache those states, and let every token read from both its online memory and the cached past — so effective memory *grows* with length and cost interpolates as `O(N·L)` between RNN (`O(L)`) and attention (`O(L²)`). But the paper's weight sits on what you build on that backbone, across **four co-equal design axes**:

- **How memories are combined / routed.** Residual, Gated Residual (GRM), Memory Soup, and Sparse Selective Caching (SSC) are four distinct memory architectures — drawing on weight-souping (model merging) and MoE routing — not minor variants.
- **What the per-segment memory *is*.** From plain linear attention up to *test-time / deep* memories: DLA (gradient-based memory) and Titans (momentum + weight-decay deep memory).
- **How history is segmented.** Constant vs logarithmic (Fenwick) segmentation — this is the knob that tunes complexity between `O(L)`, `O(L log L)`, and `O(L²)`.
- **A unifying lens.** With the right settings, **gated global attention, attention–RNN hybrids, and log-linear attention all fall out as special cases** (see [Unifying view](#unifying-view)).

So this repo is **not "one caching layer."** It implements the full design space — base rule × aggregation × segmentation × init — plus the reductions that pull existing architectures back into it. Caching is the organizing principle, not the whole contribution.

### Honest positioning

- MC **closes the gap** with Transformers on long-context recall and **outperforms same-parameter, same-token baseline RNNs / linear-attention models** (paper Tables 1–3). It does **not** beat Transformers on recall, and it is **not** a general-purpose drop-in that beats tuned small LLMs like Llama/Qwen.
- Strongest practical wins: long-context efficiency, post-training length extrapolation, and domain-specific settings.

---

## How this differs from existing projects

The linear-attention / RNN-memory open-source landscape splits into two camps: **kernel libraries** that optimize *base update rules* for speed, and **single-paper reimplementations** of one architecture. Memory Caching is neither — it is a **unifying framework spanning four design axes** (base rule × aggregation × segmentation × init), under which gated attention, attention–RNN hybrids, and log-linear attention appear as *special cases*. No existing library models this design space. That difference shows up in both the content and the name.

### Differs in content

Dates below are repo creation years (open source); the Memory Caching method itself comes from the **Feb 2026** paper (arXiv:2602.24281).

| Project | Created | Primary axis | What it gives you | MC segment-cache + aggregation framework? | Faithful to *this* paper |
|---|---|---|---|---|---|
| **`growing-memory-pytorch`** (this) | 2026 | Unifying growing-memory framework (4 axes) | 4 aggregations × 4 base rules, 2 segmentations, 2 init modes — plus architecture reductions | ✅ the entire point | ✅ enforced by equivalence tests |
| `flash-linear-attention` (fla-org) | 2024 | Hardware-efficient kernels | Many base rules (GLA, DeltaNet, GDN/-2, RWKV-7, …) as Triton/TileLang kernels | ❌ no segment-caching aggregation layer | n/a (different paper set) |
| `lucidrains/titans-pytorch` | 2025 | Readable single-paper reimpl | Titans (Memory-as-Context) only | ⚠️ Titans is *one* base rule; no MC aggregations | for Titans, not MC |
| `NX-AI/mlstm_kernels` | 2025 | mLSTM kernels | mLSTM train/infer kernels (JAX/PyTorch/Triton) | ❌ | n/a |
| `QwenLM/FlashQLA` | 2026 | GDN kernel speed | Fast GDN chunked prefill on Hopper | ❌ | n/a |

Concretely, fla and friends give you the **base update rules** (one axis of our design space); none of them give you the aggregation strategies (Residual / GRM / Memory Soup / SSC), the segmentation/init axes, or the segment-caching machinery that makes effective memory *grow*. `titans-pytorch` implements exactly one point in that space. This repo's deliverable **is** the full design space (base rule × aggregation × segmentation × init) **plus** the unification results that recover gated attention, hybrids, and log-linear attention as special cases — and the equivalence tests that prove each holds.

> Note: Titans and Memory Caching share an author (Behrouz et al.). This project is designed to *compose with*, not replace, single-rule repos — Titans / DLA / SWLA can be plugged in as the base rule under any MC aggregation.

### Differs in name

| Naming convention | Examples | What it signals |
|---|---|---|
| `flash-*`, `*-kernels` | flash-linear-attention, mlstm_kernels, FlashQLA | speed-first kernel library, many architectures |
| `<base-architecture>` | mamba, rwkv, titans | tied to one specific model |
| `<base-architecture>-pytorch` | titans-pytorch, gla-pytorch | faithful, readable single-paper reimpl |
| **`<capability>-pytorch`** (this) | **`growing-memory-pytorch`** | named after the *capability* (memory that grows), with MC as the mechanism inside — not a base architecture |

The naming choice is deliberate: nearly every repo in this space is named after a **base architecture** (mamba, rwkv, titans). Ours is named after the **capability** — *memory that grows with the sequence* — because that is the property the whole framework exists to deliver, and Memory Caching is the mechanism that achieves it. The name tells you this is a property you add *on top of* an RNN, not yet another RNN. We keep the `-pytorch` suffix to inherit the "faithful, readable reimplementation" convention (à la lucidrains) rather than the "fast kernel grab-bag" convention of `flash-*`.

---

## Install

```bash
pip install growing-memory          # import as: growing_memory
# or, from source
git clone https://github.com/baryon-labs/growing-memory-pytorch
cd growing-memory-pytorch && pip install -e .
```

Requires Python ≥ 3.10 and PyTorch ≥ 2.x.

## Quick start

```python
import torch
from growing_memory import MCSequenceModel, MCConfig

cfg = MCConfig(
    base_rule    = "titans",        # "linear" | "swla" | "dla" | "titans"
    aggregation  = "ssc",           # "residual" | "grm" | "soup" | "ssc"
    segmentation = "constant",      # "constant" | "logarithmic"
    init_mode    = "independent",   # "checkpoint" | "independent"
    segment_len  = 256,
    top_k        = 4,               # SSC only
    gate_input   = "u_proj",        # "u_proj" | "query"  (u_t in Eq. 10)
    d_model=1536, n_layers=24, n_heads=16, vocab_size=32000,
    mem_mlp_layers=2, mem_expansion=4,   # deep memory (DLA / Titans)
)

model = MCSequenceModel(cfg)
ids   = torch.randint(0, cfg.vocab_size, (2, 4096))
logits = model(ids)                 # (2, 4096, vocab_size)
```

A single MC attention layer can be used standalone:

```python
from growing_memory import MemoryCachingLinearAttention

layer = MemoryCachingLinearAttention(d_model=512, n_heads=8, segment_len=256, variant="grm")
y = layer(torch.randn(2, 1024, 512))   # (2, 1024, 512)
```

---

## Design space (the four axes)

This is the product. Every combination is reachable from config and must pass shape/gradient tests. The two tables below are the **base rule × aggregation** plane; segmentation and init are the other two axes.

| Aggregation \ Base rule | Linear | SWLA | DLA | Titans |
|---|---|---|---|---|
| **Residual** (Eq. 7) | collapses to fixed memory (EQ-1) | ✔ | ✔ | ✔ |
| **GRM** (Eq. 9–10) | ✔ | = Soup | ✔ | ✔ |
| **Memory Soup** (Eq. 14–15) | = GRM | = GRM | ✔ (diverges) | ✔ (diverges) |
| **SSC** (Eq. 16–17) | ✔ | ✔ | ✔ | ✔ |

Base update rules: Linear Attention (Eq. 12), SWLA c=2 (Eq. 28–29), DLA (Eq. 30–33), Titans (Eq. 34–36).

### Segmentation
- `constant` — equal length `C`; cost `O(p·L²/C)`.
- `logarithmic` — Fenwick-style power-of-two segments, up to `N = log₂L`; cost `O(p·L·logL)` (Sec. 4.2).

### Memory initialization (Sec. 3.4)
- `checkpoint` — each segment continues from the previous final state.
- `independent` — each segment compresses independently.

---

## Unifying view

A central claim of the paper — and the reason this is a *framework* rather than a single method — is that several existing architectures are **special cases** of memory caching. This repo exposes each reduction as a runnable configuration backed by an equivalence test, so hybrids and log-linear attention are not separate things to reimplement; they are points in the same configuration space.

| Set up MC as… | …and you recover | Reference |
|---|---|---|
| `segment_len = L` (a single segment) | a plain recurrent RNN / linear-attention model | EQ-5 |
| `segment_len = 1` + valueless vector memory | gated global softmax attention | Eq. 18–20, EQ-6 |
| compressor (`q = 1`) + a global-attention block | an attention–RNN **hybrid** layer | Sec. 4.1, EQ-7 |
| GRM + logarithmic segmentation | log-linear attention (Guo et al. 2025) — shipped as the `Log-Linear++` baseline | Sec. 4.3 |

The equivalence tests in [Correctness](#correctness) double as a map of this design space: passing them is what proves the reductions actually hold in code.

---

## Correctness

"Faithful" here means: the equivalence relations the paper derives are enforced as automated tests, run deterministically at small dimensions in CI.

| ID | Property (paper reference) | Test |
|---|---|---|
| EQ-1 | linear memory + Residual = fixed-size memory (Eq. 13) | matches plain linear-attn RNN, `atol ≤ 1e-5` |
| EQ-2 | linear memory + GRM does **not** collapse | input-dependent γ is not pre-summable (counterexample) |
| EQ-3 | Memory Soup = GRM (linear), ≠ (deep) | output match / mismatch respectively |
| EQ-4 | SSC with `k = N−1` = GRM | output match, `atol ≤ 1e-5` |
| EQ-5 | `segment_len = L` (N=1) → plain recurrent RNN | matches base rule alone |
| EQ-6 | `segment_len = 1` + valueless memory = gated global attention (Eq. 20) | matches gated softmax attention |
| EQ-7 | hybrid (compressor + attention) = checkpoint MC, segment 1 | Sec. 4.1 equivalence |
| SEG-1 | logarithmic segmentation, `L=37` → `[32, 4, 1]` | exact split |
| GRAD | DLA / Titans inner-loop optimization | analytic vs numerical gradient |

```bash
pytest tests/                # all equivalence + shape tests
pytest tests/test_equivalence.py -k EQ-1
```

---

## Reproduction status

Targets follow the paper's experimental protocol (FineWeb + Long-Data-Collections; AdamW, lr 4e-4, cosine, 0.5M-token batches, 32K vocab). Models: 760M (24 blocks, dim 1536, 16 heads, 30B tokens) and 1.3B (18 blocks, dim 2048, 8 heads, 100B tokens).

| ID | Target | Script | Status |
|---|---|---|---|
| RP-1 | Table 1 — LM ppl + commonsense | `experiments/eval_lm.py` | planned |
| RP-2 | Table 2 — NIAH (S-NIAH-1/2/3) | `experiments/eval_niah.py` | planned |
| RP-3 | Table 3 — in-context retrieval | `experiments/eval_recall.py` | planned |
| RP-4 | Table 4 — LongBench | `experiments/eval_longbench.py` | planned |
| RP-5 | Fig. 5 — MQAR | `experiments/eval_mqar.py` | **in progress** |
| RP-6 | Fig. 4 — training throughput | `experiments/bench_throughput.py` | planned |
| RP-7 | Table 5 — ablations | `experiments/ablation.py` | planned |

Success criterion (760M, reduced-token setting): Table 1 mean accuracy within **±1.0%p** of the paper, and the qualitative ordering (all MC variants > matched baseline; SSC most efficient on long inputs) reproduced. The 1.3B/100B runs are a stretch goal gated on available compute.

Reproduction logs and figures accumulate in [`docs/reproduction.md`](docs/reproduction.md).

---

## Repository layout

```
growing_memory/
├─ memory.py          # LinearAttention, SWLA, DLA, Titans (base update rules)
├─ caching.py         # segmentation, caching, retrieval (MC core, Eq. 4–5)
├─ aggregation.py     # Residual / GRM / Soup / SSC (Eq. 7, 9–10, 14–17)
├─ segmentation.py    # constant / logarithmic (Fenwick)
├─ layers.py          # block (norm, residual), gating (Eq. 10)
└─ model.py           # MCSequenceModel (LM wrapper)
configs/              # 760m.yaml, 1p3b.yaml, per-task configs
experiments/          # train.py, eval_*.py, bench_throughput.py, ablation.py
tests/                # equivalence (EQ-*), shapes, segmentation
docs/                 # equation↔code mapping, reproduction guide
```

---

## Assumptions (where the paper is underspecified)

These are explicit choices, exposed via config and revisited if the authors clarify:

- **Gating normalization.** softmax over allowed segments `{0..s}`; the online segment shares the same normalizer as cached segments.
- **MeanPooling.** GRM uses mean of per-token context projections; SSC relevance uses the sum of keys, `Σ_{j∈S} k_j` (Eq. 17). Both are swappable.
- **Connector `u_t`.** default `u_t = x_t W_u` (Eq. 10). Note: sharing `u = q` collapsed training in the paper's ablation (Table 5), so it is **off by default** and flagged.
- **Deep memory.** 2-layer MLP, expansion 4, GELU, per-chunk residual + layernorm: `M(x) = x + W₁σ(W₂x)`.
- **Feature map.** `elu(x)+1` for linear-attention base rules.

See [`docs/equation-mapping.md`](docs/equation-mapping.md) for the full equation↔code table.

---

## Roadmap

- **M0** core + Linear + Residual/GRM + tests EQ-1/2/5
- **M1** Soup + SSC (EQ-3/4), both segmentations (SEG-1)
- **M2** DLA / Titans / SWLA deep memory + GRAD test
- **M3** LM wrapper + `train.py` + RP-1 (760M, reduced)
- **M4** long-context evals RP-2..5 + Log-Linear++ baseline
- **M5** efficiency (SSC gather, chunked parallel scan) + RP-6
- **M6** docs, examples, `v0.1.0` release

---

## Citation

If you use this code, please cite the original paper:

```bibtex
@article{behrouz2026memorycaching,
  title   = {Memory Caching: RNNs with Growing Memory},
  author  = {Behrouz, Ali and others},
  journal = {arXiv preprint arXiv:2602.24281},
  year    = {2026}
}
```

To reference this implementation specifically, see `CITATION.cff`.

## Contributing

Contributions welcome. PRs that touch model behavior must (1) pass the equivalence tests, and (2) link each change to the relevant equation in `docs/equation-mapping.md`. See `CONTRIBUTING.md`.

## License

Apache-2.0. See `LICENSE`.

---

*Maintained by [Baryon Labs](https://baryon-labs.com). Unofficial reimplementation — corrections and reproduction reports are especially appreciated.*
