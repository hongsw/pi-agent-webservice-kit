"""트랙 C S1 — VibeThinker-3B AIME25 재현 (bf16, pass@1).

공식 설정(temp=1.0, top_p=0.95)으로 AIME 2025 30문항을 풀어 claim 91.4(CLR미적용)와 대조한다.
규칙기반 채점: 마지막 \\boxed{} 정수 == 정답.  avg@k는 --samples 로.

실행(4090, GPU 여유 시): ~/gm_venv/bin/python s1_aime_eval.py --max-new 16384
"""
import re, json, time, argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="WeiboAI/VibeThinker-3B")
ap.add_argument("--dataset", default="yentinglin/aime_2025")
ap.add_argument("--max-new", type=int, default=16384)
ap.add_argument("--samples", type=int, default=1)       # pass@1; >1이면 avg@k
ap.add_argument("--temperature", type=float, default=1.0)
ap.add_argument("--top-p", type=float, default=0.95)
ap.add_argument("--limit", type=int, default=0)          # 디버그용 문항 제한(0=전체)
ap.add_argument("--out", default="/home/martin/dev/c_s1_aime.json")
A = ap.parse_args()


def load_aime():
    from datasets import load_dataset
    for name in [A.dataset, "math-ai/aime25"]:
        try:
            ds = load_dataset(name)
            d = ds[list(ds.keys())[0]]
            probs = []
            for r in d:
                q = r.get("problem") or r.get("question")
                a = r.get("answer")
                if q is None or a is None:
                    continue
                probs.append((q, str(a).strip()))
            if probs:
                print(f"dataset={name} n={len(probs)}", flush=True)
                return probs
        except Exception as e:
            print("dataset fail", name, str(e)[:80], flush=True)
    raise SystemExit("no AIME dataset")


def extract_int(txt):
    m = re.findall(r"\\boxed\{([^}]*)\}", txt)
    cand = m[-1] if m else txt[-120:]
    nums = re.findall(r"-?\d+", cand.replace(",", ""))
    return int(nums[-1]) if nums else None


print("loading", A.model, "(bf16)", flush=True)
tok = AutoTokenizer.from_pretrained(A.model, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    A.model, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="cuda").eval()
print("model loaded", flush=True)

probs = load_aime()
if A.limit:
    probs = probs[:A.limit]

results = []
t0 = time.time()
for i, (q, gold) in enumerate(probs):
    try:
        gold_i = int(re.sub(r"[^\d-]", "", gold))
    except Exception:
        gold_i = None
    msgs = [{"role": "user", "content": q + "\nPlease reason step by step, and put your final answer within \\boxed{}."}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = tok(text, return_tensors="pt").to("cuda")
    correct_any = False
    samp = []
    for s in range(A.samples):
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=A.max_new, do_sample=True,
                                 temperature=A.temperature, top_p=A.top_p, top_k=None)
        gen = out[0][inp.input_ids.shape[1]:]
        pred = extract_int(tok.decode(gen, skip_special_tokens=True))
        truncated = gen.shape[0] >= A.max_new
        ok = (pred is not None and gold_i is not None and pred == gold_i)
        correct_any = correct_any or ok
        samp.append({"pred": pred, "correct": ok, "tokens": int(gen.shape[0]), "truncated": truncated})
    pass1 = samp[0]["correct"]
    avgk = sum(x["correct"] for x in samp) / len(samp)
    rec = {"idx": i, "gold": gold_i, "pass1": pass1, "avg_k": avgk,
           "preds": [x["pred"] for x in samp], "tokens": [x["tokens"] for x in samp],
           "truncated_any": any(x["truncated"] for x in samp)}
    results.append(rec)
    print(f"[{i+1}/{len(probs)}] gold={gold_i} pred={samp[0]['pred']} pass1={pass1} "
          f"avg@{A.samples}={avgk:.2f} tok={samp[0]['tokens']} trunc={rec['truncated_any']} "
          f"| running pass@1={sum(r['pass1'] for r in results)}/{len(results)}", flush=True)

n = len(results)
pass1 = sum(r["pass1"] for r in results) / n
avgk = sum(r["avg_k"] for r in results) / n
alltok = [t for r in results for t in r["tokens"]]
summary = {"model": A.model, "dataset": A.dataset, "n": n, "samples": A.samples,
           "max_new": A.max_new, "temperature": A.temperature, "top_p": A.top_p,
           "pass@1": pass1, f"avg@{A.samples}": avgk,
           "claim_no_clr": 91.4, "claim_clr": 96.7,
           "mean_tokens": sum(alltok) / len(alltok), "trunc_rate": sum(r["truncated_any"] for r in results) / n,
           "wall_sec": round(time.time() - t0, 1), "problems": results}
json.dump(summary, open(A.out, "w"), indent=2)
print("=== SUMMARY ===")
print(json.dumps({k: summary[k] for k in ["pass@1", f"avg@{A.samples}", "mean_tokens", "trunc_rate", "n", "wall_sec"]}, indent=2))
print(f"claim(CLR미적용)=91.4  우리 pass@1={pass1*100:.1f}%")
print("wrote", A.out)
