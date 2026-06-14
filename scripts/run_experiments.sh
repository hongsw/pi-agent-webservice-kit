#!/usr/bin/env bash
# AutoResearch 작은 실험 배터리 — proxy 2종 × 시드 변동 × T0(SSL고정) + Skill/lab/MCP/export.
# 4090 또는 로컬 어디서나 동작(mock 백엔드, torch 있으면 GPU 프로브 보고).
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
AR="$ROOT/tutorial/autoresearch"
RUNS="$AR/runs/experiments"
mkdir -p "$RUNS"
cd "$AR"
PY=python3

hr(){ printf '\n──────── %s ────────\n' "$1"; }
brief(){ $PY - "$1" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
b=d.get("best") or {}
print(f"  backend={d['backend']} jobs={d['jobs_run']} gate_reject={d['rejected_by_gate']} "
      f"top_rung={d['reached_top_rung']} rank_corr={d['proxy_full_rank_corr']} "
      f"best_full={b.get('full_score')} cfg={b.get('cfg',{}).get('base_rule')}+{b.get('cfg',{}).get('aggregation')}")
PY
}
run(){ # name leaderboard extra-args...
  local name="$1" lb="$2"; shift 2
  hr "$name"
  $PY run.py run --config config/run_example.yaml --leaderboard "$lb" "$@" \
     > "$RUNS/$name.summary.json" 2>"$RUNS/$name.log"
  # summary는 stdout 끝의 JSON 블록 → 별도 추출
  $PY - "$RUNS/$name.summary.json" <<'PY' 2>/dev/null || true
import json,sys,re
t=open(sys.argv[1]).read()
i=t.find("=== summary ===")
if i>=0:
    obj=json.loads(t[t.find("{",i):])
    json.dump(obj,open(sys.argv[1],"w"),ensure_ascii=False,indent=2)
PY
  brief "$RUNS/$name.summary.json"
}

echo "######## AutoResearch 실험 배터리 시작 ########"
$PY -c "import sys;sys.path.insert(0,'.');from autoresearch.model_adapter import gpu_info;import json;print('GPU:',json.dumps(gpu_info(),ensure_ascii=False))"

# E1~E3: proxy 2종 + T0
run E1_full_mqar    "$RUNS/e1.jsonl" --run-id E1_full_mqar    --proxy factory_mqar      --seed 7
run E2_full_short   "$RUNS/e2.jsonl" --run-id E2_full_short   --proxy short_horizon_pred --seed 7
run E3_t0_mqar      "$RUNS/e3.jsonl" --run-id E3_t0_mqar      --proxy factory_mqar --no-ssl --seed 7

# E4: 시드 변동(분산 점검)
for s in 1 2 3; do
  run "E4_seed$s"   "$RUNS/e4_s$s.jsonl" --run-id "E4_seed$s" --proxy factory_mqar --seed "$s"
done

# Skill: validity-gate 통과율
hr "Skill: validity-gate (space 300)"
$PY "$ROOT/skills/validity-gate/scripts/check.py" --space config/run_example.yaml -n 300

# Skill: leaderboard-analysis (E1 기준)
hr "Skill: leaderboard-analysis (E1)"
$PY "$ROOT/skills/leaderboard-analysis/scripts/analyze.py" --leaderboard "$RUNS/e1.jsonl" -n 3 \
  | $PY -c "import sys,json;d=json.load(sys.stdin);print('  rank_corr',d['proxy_full_rank_corr'],'trust',d['proxy_trust'],'best',d['recommendation']['trial_id'],'export_safe',d['recommendation']['export_safe'])"

# lab: karpathy ratchet (snapshot 모드, 2 시드)
hr "lab: ratchet (snapshot, 30 iter)"
cd "$AR/lab"; cp train.py /tmp/_train_orig.py
for s in 0 1; do
  $PY run_lab.py --iterations 30 --seed "$s" 2>/dev/null | grep -E "=== ratchet|kept|reverted|best" | tr '\n' ' '
  echo "  (seed $s)"
  cp /tmp/_train_orig.py train.py
done
rm -rf .ratchet_snapshots; cd "$AR"

# MCP selftest
hr "MCP: selftest"
$PY "$ROOT/web/mcp/autoresearch_mcp.py" --selftest 2>/dev/null | grep -E "^tools:"

# export best (E1)
hr "export best (E1)"
$PY run.py export --leaderboard "$RUNS/e1.jsonl" --out "$RUNS/export_e1" \
  | $PY -c "import sys,json;d=json.load(sys.stdin);print('  ->',d['export_path']); b=d['bundle']; print('  best full',b['full_score'],'encoder',b['encoder'],'commit',b['code_commit'])"

hr "DONE"
echo "결과: $RUNS/"
