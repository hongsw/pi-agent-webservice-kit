import sys, torch
sys.path.insert(0,'.')
from autoresearch.real.model import build_real
from compare3 import Transformer, D, H, LAYERS, VOCAB
dev="cuda"; dh=D//H
cfg={"base_rule":"linear","aggregation":"residual","segmentation":"logarithmic",
     "init_mode":"independent","segment_len":256,"d_model":D,"n_layers":LAYERS,"n_heads":H,
     "ssl":{"encoder":"none","invariance_coeff":"low"}}
lin=build_real(cfg,VOCAB,4100).to(dev).eval()
import torch.nn.functional as F
def peak(): torch.cuda.synchronize(); return torch.cuda.max_memory_allocated()/1e9
print(f"소형모델 d={D} L{LAYERS} h{H}  (KV캐시 = {2*LAYERS*D*4/1024:.0f}B/token, fp32)",flush=True)
print(f"{'context L':>10} {'TF KV캐시 peak':>15} {'선형 상태 peak':>15}",flush=True)
for L in [8192, 32768, 131072, 524288, 1048576, 2097152, 4194304]:
    # 트랜스포머: 레이어별 K,V 캐시 [1,H,L,dh] 보유 + 1 디코드 스텝
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    tf_res="?"
    try:
        with torch.no_grad():
            caches=[(torch.zeros(1,H,L,dh,device=dev),torch.zeros(1,H,L,dh,device=dev)) for _ in range(LAYERS)]
            q=torch.randn(1,H,1,dh,device=dev)
            for K,V in caches: _=F.scaled_dot_product_attention(q,K,V)
        tf_res=f"{peak():.2f}GB"
        del caches
    except RuntimeError as e:
        tf_res="OOM" if "out of memory" in str(e).lower() else "ERR"; torch.cuda.empty_cache()
    # 선형: 재귀 상태(O(1)) + 1 스텝 — L 무관
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        st=lin.init_states(1,dev,torch.float32)
        _,_=lin.step(torch.randint(1,VOCAB,(1,),device=dev),0,st)
    lin_res=f"{peak()*1000:.1f}MB"
    print(f"{L:>10} {tf_res:>15} {lin_res:>15}",flush=True)
print("done",flush=True)
