"""트랙 C R1 — VibeThinker-3B 로드 + 단일 AIME 문항 추론 (사고연쇄 + boxed 정답).

GPU 여유 시 실행:  ~/gm_venv/bin/python r1_load_vibethinker.py
"""
import re, time, argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="WeiboAI/VibeThinker-3B")
ap.add_argument("--max-new", type=int, default=8192)
ap.add_argument("--temperature", type=float, default=1.0)
ap.add_argument("--top-p", type=float, default=0.95)
A = ap.parse_args()

# 정답이 알려진 AIME 2024 I 문제 1 (정답 = 204). R1 sanity용.
PROBLEM = ("Every morning Aya goes for a 9-kilometer-long walk and stops at a coffee shop afterwards. "
           "When she walks at a constant speed of s kilometers per hour, the walk takes her 4 hours, "
           "including t minutes spent in the coffee shop. When she walks s+2 kilometers per hour, "
           "the walk takes her 2 hours and 24 minutes, including t minutes in the coffee shop. "
           "Suppose Aya walks at s+1/2 kilometers per hour. Find the number of minutes the walk takes her, "
           "including the t minutes spent in the coffee shop.")
GOLD = 204

print("loading", A.model, flush=True)
tok = AutoTokenizer.from_pretrained(A.model, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    A.model, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="cuda").eval()
nparam = sum(p.numel() for p in model.parameters())
print(f"model loaded: {nparam/1e9:.2f}B params", flush=True)

msgs = [{"role": "user", "content": PROBLEM + "\nPlease reason step by step, and put your final answer within \\boxed{}."}]
text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
inp = tok(text, return_tensors="pt").to("cuda")
t0 = time.time()
with torch.no_grad():
    out = model.generate(**inp, max_new_tokens=A.max_new, do_sample=True,
                         temperature=A.temperature, top_p=A.top_p, top_k=None)
gen = out[0][inp.input_ids.shape[1]:]
txt = tok.decode(gen, skip_special_tokens=True)
dt = time.time() - t0
ntok = gen.shape[0]

m = re.findall(r"\\boxed\{([^}]*)\}", txt)
pred = m[-1].strip() if m else None
nums = re.findall(r"-?\d+", pred) if pred else re.findall(r"-?\d+", txt[-200:])
pred_int = int(nums[-1]) if nums else None
correct = pred_int == GOLD

print("\n===== REASONING (앞 1200자) =====")
print(txt[:1200])
print("\n===== TAIL (뒤 400자) =====")
print(txt[-400:])
print(f"\n===== RESULT =====")
print(f"gold={GOLD} pred_boxed={pred!r} pred_int={pred_int} correct={correct}")
print(f"gen_tokens={ntok} time={dt:.1f}s ({ntok/dt:.1f} tok/s) max_new={A.max_new}")
print("R1_PASS" if (m and pred_int is not None) else "R1_NO_BOXED")
