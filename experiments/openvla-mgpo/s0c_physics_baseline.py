"""트랙 B S0c — OpenVLA LIBERO **물리오류 실측** baseline.

S0b의 jerk/관절한계 proxy를 넘어 sim에서 직접 물리 위반을 측정한다(② 물리오류 정교화):
  - penetration: 접촉 interpenetration 깊이(dist<0)  → 상호관통(물리 비현실)
  - arm_collision: 로봇 팔 링크(link1~7, 손가락 제외)가 물체/테이블과 충돌
  - max_contact_force: 최대 접촉 법선력(슬램)
  - drop: 물체(black bowl)를 들었다가(z↑) 떨어뜨림(z↓)
  - knock: 비파지 물체의 수평 변위(서툰 밀침)
episode physics_error = (penetration | arm_collision | drop) 중 하나라도 발생.

실행(4090): MUJOCO_GL=egl ~/openvla_venv/bin/python s0c_physics_baseline.py --tasks 10 --episodes 5
"""
import os, json, time, argparse
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
import torch
import mujoco
_orig = torch.load
torch.load = lambda *a, **k: _orig(*a, **{**k, "weights_only": False})
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

ap = argparse.ArgumentParser()
ap.add_argument("--suite", default="libero_spatial")
ap.add_argument("--tasks", type=int, default=10)
ap.add_argument("--episodes", type=int, default=5)
ap.add_argument("--max-steps", type=int, default=220)
ap.add_argument("--wait", type=int, default=10)
ap.add_argument("--pen-thresh", type=float, default=0.003)   # 3mm 이상 관통=위반
ap.add_argument("--force-thresh", type=float, default=20.0)  # 20N 이상 팔충돌력=위반
ap.add_argument("--lift-thresh", type=float, default=0.05)   # 5cm 이상 들면 '들림'
ap.add_argument("--out", default="/home/martin/dev/s0c_physics_result.json")
A = ap.parse_args()
MODEL = "openvla/openvla-7b-finetuned-" + A.suite.replace("_", "-")
DUMMY = np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32)


def prep(obs):
    img = obs["agentview_image"][::-1, ::-1]
    return Image.fromarray(np.ascontiguousarray(img)).resize((224, 224))


class Phys:
    """에피소드 동안 물리오류 신호 누적."""
    def __init__(self, env):
        self.sim = env.env.sim
        self.m = self.sim.model
        self.d = self.sim.data
        self.rawm = getattr(self.m, "_model", None)
        self.rawd = getattr(self.d, "_data", None)
        self.gname = lambda g: mujoco.mj_id2name(self.rawm, mujoco.mjtObj.mjOBJ_GEOM, g)
        # 팔 링크 geom 집합 (손가락/손은 접촉 정상이므로 제외)
        self.arm_geoms = set()
        for g in range(self.m.ngeom):
            bid = self.m.geom_bodyid[g]
            bn = mujoco.mj_id2name(self.rawm, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
            if any(bn.startswith(f"robot0_link{i}") for i in range(1, 8)):
                self.arm_geoms.add(g)
        # 물체(free joint) z 추적
        self.objs = []  # (name, qpos_z_addr, is_bowl)
        for j in range(self.m.njnt):
            if self.m.jnt_type[j] == 0:  # free
                bid = self.m.jnt_bodyid[j]
                nm = mujoco.mj_id2name(self.rawm, mujoco.mjtObj.mjOBJ_BODY, bid) or f"obj{j}"
                self.objs.append((nm, self.m.jnt_qposadr[j], "bowl" in nm))
        self.z0 = {nm: float(self.d.qpos[a + 2]) for nm, a, _ in self.objs}
        self.xy0 = {nm: self.d.qpos[a:a + 2].copy() for nm, a, _ in self.objs}
        self.max_lift = {nm: 0.0 for nm, _, _ in self.objs}
        self.pen_steps = 0
        self.arm_steps = 0
        self.max_pen = 0.0
        self.max_force = 0.0
        self.steps = 0

    def update(self):
        self.steps += 1
        d, m = self.d, self.m
        pen_hit = arm_hit = False
        f = np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            if c.dist < 0:
                self.max_pen = max(self.max_pen, -c.dist)
                if -c.dist >= A.pen_thresh:
                    pen_hit = True
            mujoco.mj_contactForce(self.rawm, self.rawd, i, f)
            fn = abs(float(f[0]))
            self.max_force = max(self.max_force, fn)
            if (c.geom1 in self.arm_geoms or c.geom2 in self.arm_geoms) and fn >= A.force_thresh:
                arm_hit = True
        self.pen_steps += int(pen_hit)
        self.arm_steps += int(arm_hit)
        for nm, a, _ in self.objs:
            self.max_lift[nm] = max(self.max_lift[nm], float(d.qpos[a + 2]) - self.z0[nm])

    def finish(self):
        d = self.d
        dropped = False
        for nm, a, is_bowl in self.objs:
            zf = float(d.qpos[a + 2]) - self.z0[nm]
            if self.max_lift[nm] >= A.lift_thresh and zf < 0.02:  # 들렸다가 다시 내려옴
                dropped = True
        # knock: 비-bowl 물체 수평변위(파지대상 아닌 것이 밀린 정도)
        knock = 0.0
        for nm, a, is_bowl in self.objs:
            disp = float(np.linalg.norm(d.qpos[a:a + 2] - self.xy0[nm]))
            knock = max(knock, disp)
        return dict(pen_rate=self.pen_steps / max(self.steps, 1), max_pen=self.max_pen,
                    arm_collision_rate=self.arm_steps / max(self.steps, 1), max_force=self.max_force,
                    dropped=dropped, max_knock=knock)


print("loading", MODEL, flush=True)
proc = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModelForVision2Seq.from_pretrained(
    MODEL, trust_remote_code=True, torch_dtype=torch.float16, low_cpu_mem_usage=True).to("cuda").eval()
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
        phys = Phys(env)
        succ = False
        steps = 0
        for s in range(A.max_steps):
            prompt = f"In: What action should the robot take to {instr.lower()}?\nOut:"
            inp = proc(prompt, prep(obs)).to("cuda", dtype=torch.float16)
            with torch.no_grad():
                act = model.predict_action(**inp, unnorm_key=A.suite, do_sample=False)
            act = np.asarray(act, dtype=np.float32).reshape(-1)[:7]
            act[-1] = -np.sign(2.0 * act[-1] - 1.0)  # gripper [0,1]->[-1,1] binarize + invert
            obs, rew, done, info = env.step(act.tolist())
            phys.update()
            steps = s + 1
            if done or rew > 0:
                succ = True
                break
        pm = phys.finish()
        env.close()
        physics_error = bool(pm["pen_rate"] > 0 or pm["arm_collision_rate"] > 0 or pm["dropped"])
        rec = dict(task=task.name, ep=ep, success=succ, steps=steps, physics_error=physics_error, **pm)
        results.append(rec)
        print(f"[{ti}.{ep}] {task.name[:36]} succ={succ} physErr={physics_error} "
              f"pen={pm['pen_rate']:.2f}(max{pm['max_pen']*1000:.1f}mm) arm={pm['arm_collision_rate']:.2f} "
              f"drop={pm['dropped']} maxF={pm['max_force']:.1f} knock={pm['max_knock']*100:.1f}cm", flush=True)

n = len(results)


def rate(key):
    return sum(1 for r in results if r[key]) / n


def avg(key):
    return sum(r[key] for r in results) / n


summary = dict(model=MODEL, suite=A.suite, n_episodes=n,
               success_rate=sum(r["success"] for r in results) / n,
               physics_error_rate=rate("physics_error"),
               components=dict(
                   penetration_rate=sum(1 for r in results if r["pen_rate"] > 0) / n,
                   arm_collision_rate=sum(1 for r in results if r["arm_collision_rate"] > 0) / n,
                   drop_rate=rate("dropped"),
                   mean_max_pen_mm=avg("max_pen") * 1000,
                   mean_max_force=avg("max_force"),
                   mean_max_knock_cm=avg("max_knock") * 100),
               thresholds=dict(pen_mm=A.pen_thresh * 1000, force_N=A.force_thresh, lift_cm=A.lift_thresh * 100),
               wall_sec=round(time.time() - t0, 1), episodes=results)
json.dump(summary, open(A.out, "w"), indent=2)
print("=== SUMMARY ===")
print(json.dumps({k: summary[k] for k in ["success_rate", "physics_error_rate", "components", "n_episodes", "wall_sec"]},
                 ensure_ascii=False, indent=2))
print("wrote", A.out)
