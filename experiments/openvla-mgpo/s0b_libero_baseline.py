"""트랙 B S0b — OpenVLA LIBERO baseline. 성공률 + 물리오류 proxy 측정.

사전등록(EXPERIMENT.md)의 verifiable reward(=task success) 신호와 physics-error proxy를 확립한다.
- 주지표: success_rate (verifiable reward)
- physics proxy: joint_limit_rate(관절한계 근접률), mean_jerk(행동 급변=물리비현실), mean_trans_norm
실행(4090): MUJOCO_GL=egl ~/openvla_venv/bin/python s0b_libero_baseline.py --tasks 5 --episodes 4
"""
import os, json, time, argparse
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
import torch
_orig = torch.load
torch.load = lambda *a, **k: _orig(*a, **{**k, "weights_only": False})  # trusted LIBERO pickles
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

ap = argparse.ArgumentParser()
ap.add_argument("--suite", default="libero_spatial")
ap.add_argument("--tasks", type=int, default=3)
ap.add_argument("--episodes", type=int, default=2)
ap.add_argument("--max-steps", type=int, default=220)
ap.add_argument("--wait", type=int, default=10)
ap.add_argument("--out", default="/home/martin/dev/s0b_baseline_result.json")
ap.add_argument("--record", action="store_true", help="에피소드를 MP4로 녹화")
ap.add_argument("--video-dir", default="/home/martin/report-srv/videos")
ap.add_argument("--record-fail-only", action="store_true", help="실패 에피소드만 저장")
A = ap.parse_args()
if A.record:
    import imageio
    os.makedirs(A.video_dir, exist_ok=True)
MODEL = "openvla/openvla-7b-finetuned-" + A.suite.replace("_", "-")
DUMMY = np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32)


def prep(obs):
    img = obs["agentview_image"][::-1, ::-1]  # 180° to match training preprocessing
    return Image.fromarray(np.ascontiguousarray(img)).resize((224, 224))


def joint_limit_rate(env):
    """robot 팔 관절이 한계 2% 이내로 근접한 비율 (관절한계 proxy)."""
    try:
        sim = env.env.sim
        m, d = sim.model, sim.data
        viol = tot = 0
        for j in range(m.njnt):
            nm = m.joint_id2name(j)
            if not nm or not nm.startswith("robot0_"):
                continue
            lo, hi = m.jnt_range[j]
            if hi <= lo:
                continue
            adr = m.jnt_qposadr[j]
            q = d.qpos[adr]
            rng = hi - lo
            tot += 1
            if q < lo + 0.02 * rng or q > hi - 0.02 * rng:
                viol += 1
        return viol / tot if tot else None
    except Exception:
        return None


print("loading", MODEL, flush=True)
proc = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModelForVision2Seq.from_pretrained(
    MODEL, trust_remote_code=True, torch_dtype=torch.float16, low_cpu_mem_usage=True
).to("cuda").eval()
print("model loaded", flush=True)

bm = benchmark.get_benchmark_dict()[A.suite]()
results = []
t0 = time.time()
for ti in range(min(A.tasks, bm.n_tasks)):
    task = bm.get_task(ti)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    instr = task.language
    inits = bm.get_task_init_states(ti)
    for ep in range(A.episodes):
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
        env.seed(ep)
        env.reset()
        env.set_init_state(inits[ep % len(inits)])
        obs = None
        for _ in range(A.wait):
            obs, _, _, _ = env.step(DUMMY)
        succ = False
        prev = None
        jerks, tnorms, jl = [], [], []
        steps = 0
        frames = []
        for s in range(A.max_steps):
            if A.record:
                frames.append(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]))
            pil = prep(obs)
            prompt = f"In: What action should the robot take to {instr.lower()}?\nOut:"
            inp = proc(prompt, pil).to("cuda", dtype=torch.float16)
            with torch.no_grad():
                act = model.predict_action(**inp, unnorm_key=A.suite, do_sample=False)
            act = np.asarray(act, dtype=np.float32).reshape(-1)[:7]
            # OpenVLA→LIBERO 그리퍼 규약 정렬 (공식 eval과 동일): [0,1]→[-1,1] 이진화 후 부호 반전
            act[-1] = np.sign(2.0 * act[-1] - 1.0)
            act[-1] = -act[-1]
            tnorms.append(float(np.linalg.norm(act[:3])))
            if prev is not None:
                jerks.append(float(np.linalg.norm(act[:6] - prev[:6])))
            prev = act
            r = joint_limit_rate(env)
            if r is not None:
                jl.append(r)
            obs, rew, done, info = env.step(act.tolist())
            steps = s + 1
            if done or rew > 0:
                succ = True
                break
        env.close()
        video = None
        if A.record and frames and (succ is False or not A.record_fail_only):
            if not (A.record_fail_only and succ):
                video = f"{A.suite}_t{ti}_ep{ep}_{'SUCCESS' if succ else 'FAIL'}.mp4"
                w = imageio.get_writer(os.path.join(A.video_dir, video), fps=20,
                                       codec="libx264", quality=8, macro_block_size=1,
                                       pixelformat="yuv420p")
                for f in frames:
                    w.append_data(f)
                w.close()
        rec = dict(task=task.name, instr=instr, ep=ep, success=succ, steps=steps,
                   mean_jerk=float(np.mean(jerks)) if jerks else None,
                   mean_trans_norm=float(np.mean(tnorms)) if tnorms else None,
                   joint_limit_rate=float(np.mean(jl)) if jl else None, video=video)
        results.append(rec)
        print(f"[{ti}.{ep}] {task.name[:42]} success={succ} steps={steps} "
              f"jerk={rec['mean_jerk']} jl={rec['joint_limit_rate']}", flush=True)

n = len(results)
sr = sum(r["success"] for r in results) / n


def avg(k):
    v = [r[k] for r in results if r[k] is not None]
    return sum(v) / len(v) if v else None


summary = dict(model=MODEL, suite=A.suite, n_episodes=n, success_rate=sr, failure_rate=1 - sr,
               physics_proxy=dict(mean_jerk=avg("mean_jerk"), mean_trans_norm=avg("mean_trans_norm"),
                                  joint_limit_rate=avg("joint_limit_rate")),
               wall_sec=round(time.time() - t0, 1), episodes=results)
json.dump(summary, open(A.out, "w"), indent=2)
print("=== SUMMARY ===")
print(json.dumps({k: summary[k] for k in ["success_rate", "failure_rate", "physics_proxy",
                                          "n_episodes", "wall_sec"]}, indent=2))
print("wrote", A.out)
