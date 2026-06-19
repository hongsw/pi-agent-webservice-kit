#!/usr/bin/env python3
"""S0 (정책 측): OpenVLA-7b를 fp16으로 로드해 더미 입력에서 행동을 예측 — 검증 루프의 정책 확인.

LIBERO 환경(보상 측)은 별도 설치. 여기선 모델이 실제 행동(7-DoF)을 내는지만 확인.
"""
import sys, torch, numpy as np
from PIL import Image

DT = torch.float16
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
try:
    from transformers import AutoModelForVision2Seq, AutoProcessor
    proc = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b", torch_dtype=DT, low_cpu_mem_usage=True, trust_remote_code=True
    ).to("cuda")
    print("loaded OpenVLA-7b (fp16). params(B)=", round(sum(p.numel() for p in vla.parameters())/1e9, 2),
          "peak_mem(GB)=", round(torch.cuda.max_memory_allocated()/1e9, 1))
    keys = list(getattr(vla, "norm_stats", {}).keys())
    print("available unnorm_keys:", keys[:8])
    unnorm = "bridge_orig" if "bridge_orig" in keys else (keys[0] if keys else None)
    img = Image.fromarray((np.random.rand(224, 224, 3) * 255).astype("uint8"))
    prompt = "In: What action should the robot take to pick up the object?\nOut:"
    inputs = proc(prompt, img).to("cuda", dtype=DT)
    action = vla.predict_action(**inputs, unnorm_key=unnorm, do_sample=False)
    print(f"predict_action OK (unnorm_key={unnorm}) → action(7-DoF)={np.round(np.asarray(action),3)}")
except Exception as e:
    import traceback; traceback.print_exc()
    print("LOAD/PREDICT FAILED:", str(e)[:120])
    sys.exit(1)
