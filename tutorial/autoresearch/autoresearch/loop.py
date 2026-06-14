"""§4 AutoResearch 루프 — 모든 조각을 엮는다.

while budget:
    cfg   = controller.sample(space)         # ASHA가 신규/승급 job 분배
    if not validity_gate(cfg): continue      # §3.4 동치/shape 게이트
    model = build(cfg)                        # growing-memory(실물) 또는 mock
    score = train_and_eval(model, proxy, rung)
    controller.report(cfg, score)            # 조기중단/승급
    leaderboard.write(...)                    # last-write-wins
best = leaderboard.top()
export(best)

karpathy ratchet과의 연결: 최상위 rung에서 full 평가를 받은 trial은 Ratchet에 제안되어
'best를 갱신할 때만 채택'된다(원본의 keep, 즉 git commit에 대응). 리더보드는 모든 시도를
기록(일반화 레이어), Ratchet은 best 추적을 담당.
"""

from __future__ import annotations

import random
from typing import Any, Callable

from .config_io import load_run_config
from .controller_asha import ASHAController, Job
from .data_interface import NASDataInterface
from .leaderboard import Leaderboard, Record
from .model_adapter import ModelHandle, backend_label, build, configure_real, gpu_info
from .proxy import evaluate_full, evaluate_proxy, rank_correlation
from .ratchet import Ratchet, current_commit
from .search_space import SearchSpace, config_hash, sample_config
from .validity_gate import validity_gate


def _make_sampler(space: SearchSpace, max_trials: int, rng: random.Random,
                  on_reject: Callable[[dict, str], None] | None = None):
    """유효성 게이트를 통과한 cfg만 내주는 샘플러. max_trials개 신규까지만 샘플."""
    state = {"emitted": 0, "seen": set()}

    def sampler() -> dict[str, Any] | None:
        if state["emitted"] >= max_trials:
            sampler._exhausted = True  # type: ignore[attr-defined]
            return None
        for _ in range(200):  # 게이트/중복 회피 재시도
            cfg = sample_config(space, rng)
            h = config_hash(cfg)
            if h in state["seen"]:
                continue
            gate = validity_gate(cfg)
            if not gate.ok:
                if on_reject:
                    on_reject(cfg, gate.reason)
                continue
            state["seen"].add(h)
            state["emitted"] += 1
            return cfg
        sampler._exhausted = True  # type: ignore[attr-defined]
        return None

    sampler._exhausted = False  # type: ignore[attr-defined]
    return sampler


def autoresearch_loop(
    run_cfg: dict[str, Any],
    leaderboard_path: str = "runs/leaderboard.jsonl",
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """run config(dict)로 한 번의 스윕을 끝까지 실행하고 요약을 반환."""
    run_id = run_cfg.get("run_id", "run")
    search = run_cfg.get("search", {})
    rungs = search.get("rungs", [1000, 4000, 16000])
    eta = int(search.get("reduction_factor", 4))
    max_trials = int(search.get("max_trials", 24))
    seed = int(run_cfg.get("seed", 0))
    proxy_task = run_cfg.get("proxy", {}).get("task", "factory_mqar")

    space = SearchSpace.from_dict(run_cfg.get("space", {}))
    rng = random.Random(seed)
    lb = Leaderboard(leaderboard_path)
    commit = current_commit(".")

    # 실물(real) 백엔드 런타임 설정 — run config의 `real` 섹션 + proxy/seed 반영
    real_cfg = dict(run_cfg.get("real", {}))
    real_cfg.setdefault("device", "cuda")
    real_cfg["task"] = proxy_task
    real_cfg["seed"] = seed
    configure_real(**real_cfg)
    backend = backend_label()

    data_cfg = run_cfg.get("data", {})
    nas = NASDataInterface(
        manifest=data_cfg.get("manifest", ""),
        cache_dir=data_cfg.get("cache", "/tmp/autoresearch_cache"),
    )
    log(f"[autoresearch] run_id={run_id} backend={backend} proxy={proxy_task} "
        f"rungs={rungs} eta={eta} max_trials={max_trials}")
    log(f"[autoresearch] NAS: {nas.cache_status()}")
    log(f"[autoresearch] GPU: {gpu_info()}")

    rejected = {"n": 0}
    sampler = _make_sampler(space, max_trials, rng,
                            on_reject=lambda c, r: rejected.__setitem__("n", rejected["n"] + 1))
    ctrl = ASHAController(sampler=sampler, rungs=rungs, reduction_factor=eta)

    handles: dict[str, ModelHandle] = {}
    ratchet = Ratchet(better="higher")  # proxy/full은 높을수록 우수
    top_rung = len(rungs) - 1
    proxy_at_top: list[float] = []
    full_at_top: list[float] = []
    jobs_run = 0

    while True:
        job = ctrl.get_job()
        if job is None:
            break
        jobs_run += 1
        model = handles.get(job.trial_id)
        if model is None:
            model = build(job.cfg)
            handles[job.trial_id] = model

        # 누적 학습: 이번 rung 목표 스텝까지 부족분만 학습(승급은 같은 모델 계속)
        delta = max(0, job.budget_steps - model.trained_steps)
        model.train(delta, rng)

        score = evaluate_proxy(proxy_task, model, rng)
        ctrl.report(job.trial_id, job.rung_idx, score)

        status = "running"
        full_score = None
        if job.rung_idx == top_rung:
            full_score = evaluate_full(model, rng)
            proxy_at_top.append(score)
            full_at_top.append(full_score)
            decision = ratchet.propose(full_score)
            status = "done"
            log(f"  [{job.trial_id}] TOP rung proxy={score:.3f} full={full_score:.3f} "
                f"-> {decision.reason} (best={ratchet.best:.3f})")
        else:
            log(f"  [{job.trial_id}] rung{job.rung_idx}(step={job.budget_steps}) "
                f"proxy={score:.3f}")

        lb.write(Record(
            run_id=run_id, trial_id=job.trial_id, cfg=job.cfg,
            proxy_score=round(score, 4),
            full_score=round(full_score, 4) if full_score is not None else None,
            cost=float(model.trained_steps), rung=job.rung_idx, status=status,
            backend=model.backend, seed=seed, code_commit=commit,
        ))

    # §5 프록시↔full 순위상관 보정 점검
    corr = rank_correlation(proxy_at_top, full_at_top) if len(proxy_at_top) >= 2 else None
    best = lb.top(1)
    summary = {
        "run_id": run_id,
        "backend": backend,
        "jobs_run": jobs_run,
        "rejected_by_gate": rejected["n"],
        "reached_top_rung": len(full_at_top),
        "proxy_full_rank_corr": round(corr, 3) if corr is not None else None,
        "ratchet": ratchet.stats(),
        "best": _record_brief(best[0]) if best else None,
        "leaderboard": leaderboard_path,
    }
    log(f"[autoresearch] done. {summary['reached_top_rung']} reached top rung, "
        f"rank_corr={summary['proxy_full_rank_corr']}, "
        f"best full={summary['best']['full_score'] if summary['best'] else None}")
    return summary


def _record_brief(r: Record) -> dict[str, Any]:
    return {
        "trial_id": r.trial_id,
        "cfg": r.cfg,
        "proxy_score": r.proxy_score,
        "full_score": r.full_score,
        "cost": r.cost,
        "backend": r.backend,
    }


def run_from_file(config_path: str, leaderboard_path: str = "runs/leaderboard.jsonl",
                  log: Callable[[str], None] = print) -> dict[str, Any]:
    return autoresearch_loop(load_run_config(config_path), leaderboard_path, log)
