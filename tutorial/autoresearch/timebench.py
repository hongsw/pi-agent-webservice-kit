import sys, time, torch
sys.path.insert(0,'.')
from autoresearch.real.model import build_real
from compare3 import Transformer, D, H, LAYERS, VOCAB
dev="cuda"
cfg={"base_rule":"linear","aggregation":"residual","segmentation":"logarithmic",
     "init_mode":"independent","segment_len":256,"d_model":D,"n_layers":LAYERS,"n_heads":H,
     "ssl":{"encoder":"none","invariance_coeff":"low"}}
lin=build_real(cfg,VOCAB,140000).to(dev).eval()
tf=Transformer(VOCAB,D,H,LAYERS,140000).to(dev).eval()
def timed(fn,iters=5):
    with torch.no_grad():
        fn(); torch.cuda.synchronize(); t0=time.time()
        for _ in range(iters): fn()
        torch.cuda.synchronize(); return (time.time()-t0)/iters*1000
print(f"{'L':>7} {'TF(flash) ms':>13} {'Lin(chunk) ms':>14} {'TF/Lin':>7}",flush=True)
for L in [2048,8192,32768,65536,131072]:
    x=torch.randint(1,VOCAB,(1,L),device=dev)
    try: t_tf=timed(lambda: tf(x))
    except RuntimeError: t_tf=float('nan'); torch.cuda.empty_cache()
    try: t_lin=timed(lambda: lin.forward_chunked(x))
    except RuntimeError: t_lin=float('nan'); torch.cuda.empty_cache()
    print(f"{L:>7} {t_tf:>13.1f} {t_lin:>14.1f} {t_tf/t_lin:>7.2f}",flush=True)
