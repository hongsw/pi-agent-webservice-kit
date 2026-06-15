import sys, time, torch
import torch.nn.functional as F
sys.path.insert(0,'.')
from autoresearch.real.model import build_real
from autoresearch.real.data import make_mqar_batch
cfg={"base_rule":"linear","aggregation":"residual","segmentation":"logarithmic",
     "init_mode":"independent","segment_len":256,"d_model":256,"n_layers":4,"n_heads":2,
     "ssl":{"encoder":"none","invariance_coeff":"low"}}
dev="cuda"; B,V,P=1,32,8
tot=torch.cuda.get_device_properties(0).total_memory/1e9
print(f"GPU total={tot:.1f}GB  batch={B} d=256 L4 h2 chunk=256",flush=True)
m=None
for L in [4096,8192,16384,32768,65536,131072,262144,524288]:
    try:
        # 모델 최대길이 재생성(pos 임베딩 길이)
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        m=build_real(cfg,V,L+8).to(dev)
        opt=torch.optim.AdamW(m.parameters(),lr=3e-3)
        gen=torch.Generator().manual_seed(0)
        inp,tgt,_=make_mqar_batch(B,L,P,V,dev,gen)
        t0=time.time()
        logits,aux=m.forward_chunked(inp,chunk=256,return_aux=True)
        loss=F.cross_entropy(logits.reshape(-1,V),tgt.reshape(-1),ignore_index=-100)+aux
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        torch.cuda.synchronize()
        peak=torch.cuda.max_memory_allocated()/1e9
        print(f"L={L:>7}: OK  peak={peak:5.2f}GB  step={time.time()-t0:5.1f}s",flush=True)
        del m,opt,logits,loss,inp,tgt
    except RuntimeError as e:
        msg="OOM" if "out of memory" in str(e).lower() else str(e)[:50]
        print(f"L={L:>7}: {msg}",flush=True)
        torch.cuda.empty_cache(); break
print("done",flush=True)
