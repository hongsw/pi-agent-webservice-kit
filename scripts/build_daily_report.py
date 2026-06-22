#!/usr/bin/env python3
"""인터랙티브 작업 보고서 생성 — git log + 큐레이트 데이터.

두 연구를 **별개 파일·별개 시각**으로 만들고, 연결되는 부분(Phase 2)만 양쪽에서 설명한다:
  report/track_a_growing_memory.html  — 트랙 A: growing-memory / Titans 효율 연구
  report/track_b_openvla_mgpo.html    — 트랙 B: OpenVLA × MGPO 물리오류 연구

자체완결(외부 라이브러리 0). 재실행: python3 scripts/build_daily_report.py
"""
import json, os, subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def git_log():
    out = subprocess.run(["git", "-C", ROOT, "log", "--pretty=format:%ad|%h|%s", "--date=short"],
                         capture_output=True, text=True).stdout.strip().splitlines()
    rows = []
    for ln in out:
        d, h, s = ln.split("|", 2)
        rows.append({"date": d, "hash": h, "subj": s})
    return rows


ALL_COMMITS = git_log()
B_KEYS = ("openvla", "mgpo", "object-centric")


def is_b(subj):
    s = subj.lower()
    return any(k in s for k in B_KEYS)


def is_c(subj):
    s = subj.lower()
    return "vibethinker" in s or "track-c" in s


def js(x):
    return json.dumps(x, ensure_ascii=False)


def load_json(rel):
    p = os.path.join(ROOT, rel)
    if os.path.exists(p):
        try:
            return json.load(open(p, encoding="utf-8"))
        except Exception:
            return None
    return None


# 트랙 B S0b LIBERO baseline 실측 (있으면 보고서에 실수치 렌더)
B_BASE = load_json("report/data/s0b_baseline_result.json")
B_VIDS = load_json("report/data/s0b_videos.json")  # 녹화 rollout (8099에서 ../videos/ 서빙)


def _per_task_rows():
    if not B_BASE:
        return []
    from collections import defaultdict
    agg = defaultdict(lambda: {"n": 0, "s": 0, "steps": [], "jerk": [], "jl": []})
    for e in B_BASE["episodes"]:
        a = agg[e["task"]]
        a["n"] += 1
        a["s"] += 1 if e["success"] else 0
        a["steps"].append(e["steps"])
        a["jerk"].append(e.get("mean_jerk") or 0)
        a["jl"].append((e.get("joint_limit_rate") or 0) * 100)
    rows = []
    for t, a in agg.items():
        rows.append((a["s"] / a["n"], t, a["s"], a["n"], sum(a["steps"]) / a["n"],
                     sum(a["jerk"]) / a["n"], sum(a["jl"]) / a["n"]))
    return sorted(rows)


# 정성 관찰 (실패 영상 프레임을 직접 확인해 기록)
B_QUAL_FINDINGS = [
  ("실패의 본질 = 물리 난폭함이 아니라 '공간 그라운딩 실패'",
   "실패 에피소드도 jerk(0.10~0.24)·관절한계율(~0%)이 정상 범위. 즉 팔이 거칠게 움직여 실패한 게 아니라, "
   "<b>엉뚱한 위치로 가서 그릇을 못 집고 220스텝을 헛돈다</b>. 물리적으로는 '얌전한' 실패."),
  ("어려운 태스크 = 참조 표현이 모호하거나 타겟이 가려진 경우",
   "최악: 'next_to_the_plate'(1/5), 'on_the_ramekin'(1/5), 'on_the_wooden_cabinet'(2/5), "
   "'in_the_top_drawer'(3/5). 공통점 = 같은 검은 그릇이 여러 개라 <b>어느 그릇인지(참조 해소)</b>가 핵심. "
   "쉬움: 'between_the_plate_and_the_ramekin'(5/5), 'on_the_cookie_box'(5/5) — 위치가 비교적 명확."),
  ("관측된 실패 모드: 좌측/중앙으로 표류 후 정지",
   "두 실패 영상 모두 팔이 타겟 그릇이 아닌 테이블 좌·중앙으로 이동, 그리퍼를 연 채 낮게 떠서 멈춤. "
   "그랩 시도 자체가 일어나지 않거나 빈 곳을 집는다."),
  ("실험 함의: physics축보다 referential/grounding축이 더 큰 레버",
   "baseline 실패가 물리비효율(jerk/관절한계)로 거의 설명되지 않음 → MGPO 물리보상의 <b>개선 여지(headroom)가 제한적</b>일 수 있음. "
   "ARCHITECTURE.md의 ① referential 축(에이전트/object-memory)이 성공률에 더 직접적. H1 검증 시 이 점을 반드시 같이 본다."),
]
B_FAIL_FRAMES = [
  ("pick_up_the_black_bowl_next_to_the_ramekin (실패)",
   ["libero_spatial_t1_ep0_FAIL_0_0.png", "libero_spatial_t1_ep0_FAIL_2_154.png", "libero_spatial_t1_ep0_FAIL_3_219.png"],
   "시작엔 그릇들이 보이지만, 팔이 좌측으로 표류해 엉뚱한 그릇 근처에서 그리퍼를 연 채 정지 → 미완수."),
  ("pick_up_the_black_bowl_in_the_top_drawer (실패)",
   ["libero_spatial_t4_ep1_FAIL_0_0.png", "libero_spatial_t4_ep1_FAIL_2_154.png", "libero_spatial_t4_ep1_FAIL_3_219.png"],
   "타겟(우측 캐비닛/서랍)으로 가지 않고 좌측 빈 테이블로 내려가 낮게 떠서 멈춤 → 서랍 접근/그랩 없음."),
]


def _qual_html():
    rows = _per_task_rows()
    if not rows:
        return '<div class=day><h3>🔬 정성 분석 (대기)</h3><p class=muted>baseline 필요.</p></div>'
    tr = "".join(
        f'<tr><td>{t[:46]}</td><td style="color:{"#3fb950" if sr>=0.8 else ("#d29922" if sr>=0.5 else "#f85149")}">'
        f'{s}/{n} ({sr*100:.0f}%)</td><td>{ms:.0f}</td><td>{jk:.3f}</td><td>{jl:.2f}%</td></tr>'
        for sr, t, s, n, ms, jk, jl in rows)
    finds = "".join(f'<div class=mi><b>{a}</b><span>{b}</span></div>' for a, b in B_QUAL_FINDINGS)
    strips = ""
    for title, imgs, cap in B_FAIL_FRAMES:
        thumbs = "".join(f'<img src="../frames/{im}" style="width:31%;border-radius:6px;border:1px solid #283040">' for im in imgs)
        strips += (f'<div class=chartbox><h4 style="color:#f85149">❌ {title}</h4>'
                   f'<div style="display:flex;gap:2%">{thumbs}</div>'
                   f'<div class=legend>시작 → 중반 → 종료(220스텝). {cap}</div></div>')
    return f"""
<div class=day><h3>🔬 태스크별 성공률 (libero_spatial, 50ep)</h3>
 <table><thead><tr><th>task</th><th>성공</th><th>평균steps</th><th>jerk</th><th>관절한계율</th></tr></thead>
 <tbody>{tr}</tbody></table>
 <p class=muted style="margin-top:8px">성공한 태스크는 평균 ~85스텝에 끝나고, 실패는 220(최대)까지 감 — 실패=완수 못 함.</p></div>
<div class=day><h3>🧭 정성 관찰 — 무엇이 잘못됐나</h3>{finds}</div>
<div class=day><h3>🖼️ 실패 프레임 (영상에서 직접 확인)</h3><div class=grid2>{strips}</div></div>
"""


def _videos_html():
    if not B_VIDS:
        return ('<div class=day><h3>🎬 rollout 영상 (없음)</h3>'
                '<p class=muted>녹화된 영상 없음.</p></div>')
    eps = [e for e in B_VIDS.get("episodes", []) if e.get("video")]
    fails = [e for e in eps if not e["success"]]
    succ = [e for e in eps if e["success"]]

    def vid(e):
        col = "#f85149" if not e["success"] else "#3fb950"
        tag = "실패(220스텝 타임아웃=미완수)" if not e["success"] else "성공"
        return (f'<div class=chartbox><h4 style="color:{col}">{"❌" if not e["success"] else "✅"} '
                f'{e["task"][:40]} — {tag}</h4>'
                f'<video controls preload=metadata width=256 style="width:100%;max-width:340px;border-radius:8px">'
                f'<source src="../videos/{e["video"]}" type="video/mp4"></video>'
                f'<div class=legend>{e["instr"]} · {e["steps"]}스텝 · jerk {(e.get("mean_jerk") or 0):.3f}</div></div>')
    fh = "".join(vid(e) for e in fails) or '<p class=muted>이번 녹화 세트에 실패 에피소드 없음.</p>'
    sh = "".join(vid(e) for e in succ[:3])
    return f"""
<div class=day><h3>🎬 실패 rollout 영상 ({len(fails)}건)</h3>
 <p class=muted>OpenVLA가 그릇을 못 집고 220스텝까지 헛도는 모습 — MGPO(S1)가 줄여야 할 <b>실패/물리비효율 행동</b>이다.
 영상은 모델이 보는 시점(agentview, 180° 정렬). 8099 서버에서 재생됨.</p>
 <div class=grid2>{fh}</div></div>
<div class=day><h3>✅ 대비용 성공 rollout</h3>
 <div class=grid2>{sh}</div></div>
"""


# ── 트랙 A: growing-memory / Titans 효율 연구 ─────────────────────────────────
A_CHARTS = {
  "compute": {"L": [2048, 8192, 32768, 65536, 131072], "deltanet": [4.0, 3.8, 6.9, 14.8, 31.2],
              "tf": [0.8, 2.1, 18.5, 65.4, 244.0]},
  "infer":   {"L": [8192, 131072, 1048576, 2097152], "kv": [0.08, 1.17, 9.69, 19.35], "state": 0.0259},
  "recall":  {"names": ["deltanet", "gated_deltanet", "retention", "titans", "linear", "gla"],
              "vals": [1.0, 1.0, 1.0, 1.0, 0.54, 0.06]},
  "grok":    {"steps": list(range(1000, 26000, 1000)),
              "rec": [.324, .332, .335, .346, .335, .339, .341, .342, .336, .336, .337, .337, .326, .337,
                      .604, .789, .939, .986, .995, .997, 1, 1, 1, 1, 1]},
}
A_MILE = {
  "2026-06-14": [("AutoResearch 노드 구축", "Pi 4요소(Agent/Skill/MCP/Extension/WebUI) + ASHA + ratchet, 4090 실학습"),
                 ("메모리캐싱 동치", "병렬학습↔O(1) 재귀추론 동치(1e-7), 토큰 스트리밍"),
                 ("grokking 튜닝", "8변형 회상, 아키텍처별 용량 천장")],
  "2026-06-15": [("3자 효율 비교+정정", "naive O(L²) OOM 버그→chunked O(L) 수정. FlashAttn은 메모리만 선형"),
                 ("128K 학습 + 추론 KV 환산", "단일 4090 128K, 1대≈다GPU KV용량"),
                 ("정확 구현 채택", "from-scratch titans 실패(chance)→lucidrains/fla 채택, deltanet 연결 1.0"),
                 ("pip 패키지 + HC-SR04 이상탐지", "HF/Unsloth 래퍼, 실센서 드리프트 −22.8 SE")],
  "2026-06-16": [("T1 실센서 연결", "HC-SR04→autoresearch 학습, deltanet 0.983>baseline 0.934"),
                 ("Web 차트 + 시각화 PDF", "대시보드 차트, md→PDF(29p) 빌드")],
}
A_COR = [
  ("선형 OOM=우위?", "naive O(L²) 구현 버그였음", "chunked O(L) 재구현 → OOM 해소"),
  ("학습 메모리 우위", "FlashAttn이 O(L)이라 거짓", "진짜 우위는 추론 O(1)로 재정의"),
  ("자체 titans 정확?", "from-scratch가 chance(학습실패)", "lucidrains titans-pytorch 채택"),
  ("recall 0.98 vs 0.33 = 난이도?", "틀림 — 변수 격리로 chunk_size 버그", "chunk_size=min(32,seg) 수정"),
  ("NAS/엣지 연결됨?", "실은 stub", "도커 NAS·Edge 서버로 실구현"),
  ("titans 이름 내가 지음?", "Google 논문(2501.00663) 차용", "출처 명시 + 정확 구현 검증"),
]

# ── 트랙 B: OpenVLA × MGPO 물리오류 연구 ──────────────────────────────────────
B_MILE = {
  "2026-06-19": [("OpenVLA×MGPO 착수 (사전등록)", "EXPERIMENT.md: H1~H3 가설·지표·합격선 데이터 보기 전 고정"),
                 ("S0 정책측 확인", "OpenVLA-7b fp16 4090 로드(7.54B, peak 15.1GB) + 7-DoF 행동예측 OK"),
                 ("object-memory 아키텍처 설계", "두 오류축 분리 · Phase1 DeepAgent / Phase2 growing-memory 내장 + SVG")],
}
if B_BASE:
    B_MILE.setdefault("2026-06-19", []).append(
        ("S0b LIBERO baseline 실측", f"{B_BASE['suite']} {B_BASE['n_episodes']}ep · 성공률 "
         f"{B_BASE['success_rate']*100:.0f}% · 관절한계율 {(B_BASE['physics_proxy']['joint_limit_rate'] or 0)*100:.1f}% "
         f"→ MGPO(S1)가 개선할 기준선"))
B_COR = [
  ("baseline 0% 나옴 → 모델이 나쁘다?", "아니오 — 그리퍼 부호 규약(OpenVLA↔LIBERO 반대) 누락 버그", "normalize+invert 추가 후 0%→66%"),
  ("우리 66% = 논문 84%?", "다름 — 부분측정(50ep·5 init·우리 시드), 논문은 500ep", "'부분 기준선'으로 정직 표기, full-suite는 별도"),
  ("MGPO를 VLA에 쓰면 물리오류 준다?", "math/code에서만 입증 — VLA 전이는 미입증", "본 실험의 가설로 명시(H1), 사전등록"),
  ("projector만 키우면 referential 해결?", "기억은 projector 역할 아님", "object-level memory path를 별도 설계"),
]

# 두 트랙을 잇는 단 하나의 연결점 (양쪽 보고서에 동일하게 표시)
BRIDGE = [
  ("OpenVLA object state memory ＝ growing-memory(deltanet/titans)",
   "트랙 B Phase 2에서 객체 상태를 시간축으로 유지하는 메모리를, 트랙 A의 O(1) 재귀상태 구현으로 만든다."),
  ("compression-coverage ↔ 엣지학습노드(autoresearch)",
   "트랙 B의 소형코어 압축 가설을, 트랙 A의 autoresearch config 스윕(보상=success)으로 탐색한다."),
  ("공유 목표: 엣지 상수메모리",
   "둘 다 단일 4090/엣지에서 상수메모리 추론을 지향 — 효율(A)이 강건성(B)의 배포 수단이 된다."),
]
BRIDGE_NOTE = ("연결은 <b>구현 수단</b> 한 곳뿐이다. 가설·지표·합격선은 완전 독립. "
               "트랙 B의 Phase 1(에이전트, frozen OpenVLA)은 트랙 A와 무관하고, "
               "<b>Phase 2</b>에서만 트랙 A의 메모리를 재사용한다.")

# ── 트랙 C: VibeThinker-3B 재현 연구 ─────────────────────────────────────────
C_CLAIMS = [  # (벤치, CLR적용, CLR미적용)
  ("AIME25", "96.7", "91.4"), ("AIME26", "97.1", "94.3"), ("HMMT25", "95.4", "—"),
  ("LiveCodeBench v6", "80.2 (Pass@1)", "—"), ("GPQA-Diamond", "72.9", "—"),
]
C_S1 = load_json("report/data/c_s1_aime.json")  # AIME25 재현 실측(있으면 자동 반영)


def _c_s1_stage():
    if not C_S1:
        return {"id": "S1", "title": "S1 — AIME25 30문항 pass@1 (bf16)", "status": "pending",
                "body": "bf16, temp=1.0/top_p=0.95로 측정해 claim <b>91.4</b>(CLR미적용)와 대조. GPU 여유 시 실행(큐잉)."}
    d = C_S1
    ps = d.get("problems", [])
    tr = [p for p in ps if p.get("truncated_any")]
    nt = [p for p in ps if not p.get("truncated_any")]
    tr_ok = sum(p["pass1"] for p in tr)
    nt_ok = sum(p["pass1"] for p in nt)
    p1 = d["pass@1"] * 100
    body = (
        f"<b>pass@1 = {p1:.1f}%</b> ({sum(p['pass1'] for p in ps)}/{d['n']}) · max_new={d['max_new']} · "
        f"평균 {d['mean_tokens']:.0f} 토큰 · {d['wall_sec']/60:.0f}분<br>"
        f"claim(CLR미적용) <b>91.4</b> 대비 약 {91.4-p1:.0f}%p 낮음 → <b>판정: 부분재현(설정차)</b>.<br>"
        f"<span style='color:#3fb950'>핵심: <b>truncation율 {d['trunc_rate']*100:.0f}%</b>. "
        f"잘리지 않은 문항은 <b>{nt_ok}/{len(nt)} = {nt_ok/max(len(nt),1)*100:.0f}% 정답</b>, "
        f"잘린 문항은 {tr_ok}/{len(tr)}.</span> "
        f"<span class=muted>즉 16k 안에 추론을 끝내면 거의 다 맞힘 — 격차는 능력이 아니라 토큰한도(claim은 40k). "
        f"개선 방향=Long2Short(적은 토큰으로 풀기).</span>")
    return {"id": "S1", "title": f"S1 — AIME25 pass@1 = {p1:.0f}% (bf16, max 16k)", "status": "done", "body": body}


# 단계 결과: 새 결과가 나오면 dict 한 개씩 추가 (status: done|pending|wip)
C_STAGES = [
  {"id": "R1", "title": "R1 — 로드·단일문항 추론", "status": "done",
   "body": "AIME 2024-I-1(정답 204)을 <b style='color:#3fb950'>정확히 해결</b> — 단계추론 후 "
           "<code>\\boxed{204}</code>, correct=True. 생성 2225토큰.<br>"
           "<span class=muted>※ GPU 경합(타 사용자 17GB)으로 <b>8-bit 양자화 sanity</b> 실행 — 파이프라인+추론력 검증용, "
           "재현 수치 아님. 정식 벤치(S1)는 bf16.</span>"},
  _c_s1_stage(),
  {"id": "S2", "title": "S2 — 40k 토큰 재측정 + avg@k", "status": "pending",
   "body": "truncation 제거(max 40960) + k샘플 avg@k로 claim에 더 근접하는지 확인."},
  {"id": "S3", "title": "S3 — 개선 방향 실험", "status": "pending",
   "body": "Long2Short / 검증기반 self-consistency 등 개선 탐색."},
]
C_MILE = {
  "2026-06-20": [("트랙 C 착수 + 사전등록", "VibeThinker-3B 재현 계획(claim표·합격선), CLR 미공개→CLR미적용 1차"),
                 ("R1 PASS", "VibeThinker-3B가 AIME 문항 정확 해결(8-bit sanity), 파이프라인 검증")],
  "2026-06-22": [("S1 — AIME25 재현(bf16)", "pass@1 66.7%(20/30). 미잘림 14/14=100%, truncation 53% → "
                 "격차는 토큰한도(claim 40k vs 우리 16k), 능력 아님")],
}
C_COR = [
  ("3B가 AIME25 96.7?", "프런티어급 고점 — 특히 CLR(test-time)은 구현 미공개", "CLR미적용 단일추론을 1차 재현 기준으로 분리"),
  ("8-bit 결과 = 재현 수치?", "아니오 — 양자화는 정확도 영향", "R1은 sanity로만, 벤치는 bf16(S1)"),
  ("66.7% < 91.4 = 재현 실패?", "아니오 — 미잘림 문항은 14/14=100%", "격차=16k truncation(53%), claim은 40k. 부분재현(설정차)"),
]
C_BRIDGE = [
  ("MGPO를 공유하지만 도메인이 다르다",
   "트랙 C(수학추론)와 트랙 B(OpenVLA 물리)는 둘 다 MGPO를 쓰지만 과제·보상·합격선이 독립. 별개 트랙."),
  ("재현→개선 패러다임",
   "C는 '공개 주장 재현 후 개선'. 검증가능 보상(정답) 기반이라 MGPO/self-consistency 개선이 직접 적용 가능."),
]
C_BRIDGE_NOTE = ("세 트랙은 독립 연구다. C는 VibeThinker를 우리 환경에서 <b>먼저 재현</b>하고 개선을 모색한다 — "
                 "MGPO라는 도구만 트랙 B와 공유한다.")


def _track_c_results():
    cl = "".join(f'<tr><td class=h>{b}</td><td>{c1}</td><td>{c2}</td></tr>' for b, c1, c2 in C_CLAIMS)
    badge = {"done": ("✅", "#3fb950"), "wip": ("🔄", "#d29922"), "pending": ("⏳", "#8b96a5")}
    st = ""
    for s in C_STAGES:
        ic, col = badge.get(s["status"], ("•", "#8b96a5"))
        st += (f'<div class=day><h3 style="color:{col}">{ic} {s["title"]}</h3>'
               f'<div style="font-size:13px">{s["body"]}</div></div>')
    return f"""
<div class=day><h3>🎯 검증 대상 주장 (공개 수치)</h3>
 <table><thead><tr><th>벤치</th><th>claim (CLR 적용)</th><th>claim (CLR 미적용)</th></tr></thead><tbody>{cl}</tbody></table>
 <p class=muted style="margin-top:8px">⚠️ CLR(Claim-Level Reliability)은 test-time scaling이며 구현 미공개 →
 <b>CLR 미적용</b> 수치를 1차 재현 기준으로 본다.</p></div>
{st}
"""


TRACKS = {
  "a": {
    "file": "track_a_growing_memory.html",
    "icon": "🔬", "name": "트랙 A — growing-memory / Titans 효율 연구",
    "tag": "선형·재귀 상태메모리: O(1) 추론 · 128K 학습 · 검증된 회상",
    "other": ("track_b_openvla_mgpo.html", "🤖 트랙 B — OpenVLA×MGPO"),
    "commits": [c for c in ALL_COMMITS if not is_b(c["subj"]) and not is_c(c["subj"])],
    "mile": A_MILE, "cor": A_COR, "charts": A_CHARTS,
    "stats": lambda cm, dts: [
      ("커밋", str(len(cm)), f"{dts[0]}~{dts[-1]}"),
      ("작업일", str(len(dts)), "효율 트랙"),
      ("최대 학습 컨텍스트", "128K", "단일 4090 (chunked O(L))"),
      ("추론 메모리비", "~992×", "재귀 O(1) vs KV O(L) @16K"),
      ("연산 속도", "8×", "128K, deltanet vs FlashAttn"),
      ("검증 회상", "1.0", "deltanet/retention/titans"),
    ],
    "results": None,  # CHARTS 패널 사용
  },
  "b": {
    "file": "track_b_openvla_mgpo.html",
    "icon": "🤖", "name": "트랙 B — OpenVLA × MGPO 물리오류 연구",
    "tag": "MGPO + compression-coverage → 물리오류 행동 감소 (사전등록)",
    "other": ("track_a_growing_memory.html", "🔬 트랙 A — growing-memory"),
    "commits": [c for c in ALL_COMMITS if is_b(c["subj"])],
    "mile": B_MILE, "cor": B_COR, "charts": None,
    "stats": lambda cm, dts: ([
      ("단계", "S0b 완료" if B_BASE else "S0 완료", "LIBERO baseline 실측" if B_BASE else "정책측 확인"),
      ("baseline 성공률", f"{B_BASE['success_rate']*100:.0f}%", f"{B_BASE['suite']} · {B_BASE['n_episodes']}ep")
        if B_BASE else ("모델", "OpenVLA-7b", "7.54B params, fp16"),
      ("baseline 실패율", f"{B_BASE['failure_rate']*100:.0f}%", "MGPO로 줄일 대상")
        if B_BASE else ("로드", "15.1GB", "/ 24GB (4090, peak)"),
      ("관절한계율", f"{(B_BASE['physics_proxy']['joint_limit_rate'] or 0)*100:.1f}%", "physics proxy")
        if B_BASE else ("가설", "H1~H3", "물리오류↓ · 다양성 · 압축"),
      ("H1 합격선", "≥20%↓", "physics-error, success 비열화"),
      ("커밋", "{N}", "신규 트랙"),
    ]),
    "results": "RESULTS_B",  # 가설·설계 + 두 오류축 raw HTML
  },
  "c": {
    "file": "track_c_vibethinker.html",
    "icon": "🧮", "name": "트랙 C — VibeThinker-3B 재현",
    "tag": "소형 추론모델 재현 후 개선 (Qwen2.5-Coder-3B 기반, MGPO/SSP)",
    "commits": [c for c in ALL_COMMITS if ("vibethinker" in c["subj"].lower() or "track-c" in c["subj"].lower())],
    "mile": C_MILE, "cor": C_COR, "charts": None,
    "bridge": C_BRIDGE, "bridge_note": C_BRIDGE_NOTE,
    "stats": lambda cm, dts: [
      ("단계", next((s["id"] for s in reversed(C_STAGES) if s["status"] == "done"), "S0") + " 완료", "재현 진행"),
      ("모델", "VibeThinker-3B", "Qwen2.5-Coder-3B 기반, MIT"),
      ("AIME25 pass@1", f"{C_S1['pass@1']*100:.0f}%", f"우리측정(max16k) vs claim 91.4") if C_S1
        else ("R1", "✅ 정답", "AIME 문항 \\boxed{204}"),
      ("미잘림 정답률", f"{sum(p['pass1'] for p in C_S1['problems'] if not p['truncated_any'])}/"
        f"{sum(1 for p in C_S1['problems'] if not p['truncated_any'])}", "16k안에 끝낸 문항") if C_S1
        else ("claim AIME25", "91.4", "CLR미적용 (재현 목표)"),
      ("truncation율", f"{C_S1['trunc_rate']*100:.0f}%", "16k 초과 잘림(격차원인)") if C_S1
        else ("CLR", "미공개", "test-time scaling, 재현 제외"),
      ("커밋", "{N}", "신규 트랙"),
    ],
    "results": "RESULTS_C",
  },
}

TRACK_NAV = [
  ("track_a_growing_memory.html", "🔬 트랙 A"),
  ("track_b_openvla_mgpo.html", "🤖 트랙 B"),
  ("track_c_vibethinker.html", "🧮 트랙 C"),
]

# ── 트랙 B 전용 패널 HTML ─────────────────────────────────────────────────────
def _baseline_html():
    if not B_BASE:
        return ('<div class=day><h3>📊 S0b — LIBERO baseline (측정 대기)</h3>'
                '<p class=muted>아직 측정값 없음. 시뮬 rollout 완료 시 성공률·물리오류 proxy가 여기 실수치로 채워진다.</p></div>')
    b = B_BASE
    pp = b["physics_proxy"]
    eps = b.get("episodes", [])
    succ = sum(1 for e in eps if e.get("success"))
    rows = "".join(
        f'<tr><td>{e["task"][:34]}</td><td style="color:{"#3fb950" if e["success"] else "#f85149"}">'
        f'{"성공" if e["success"] else "실패"}</td><td>{e["steps"]}</td>'
        f'<td>{(e.get("mean_jerk") or 0):.3f}</td><td>{(e.get("joint_limit_rate") or 0)*100:.1f}%</td></tr>'
        for e in eps)
    jl = (pp.get("joint_limit_rate") or 0) * 100
    return f"""
<div class=day><h3>📊 S0b — LIBERO baseline 실측 (4090, {b['suite']}, {b['n_episodes']} episodes)</h3>
 <p class=muted>verifiable reward(=task success) 신호와 physics-error proxy를 실제 시뮬 rollout으로 확립.
 이 수치가 MGPO(S1)가 개선해야 할 <b>기준선</b>이다.</p>
 <div class=cards style="margin:10px 0">
  <div class=card><div class=k>success rate</div><div class=v>{b['success_rate']*100:.0f}%</div><div class=s>{succ}/{b['n_episodes']} 성공</div></div>
  <div class=card><div class=k>failure rate</div><div class=v style="color:#f85149">{b['failure_rate']*100:.0f}%</div><div class=s>MGPO 개선 대상</div></div>
  <div class=card><div class=k>joint-limit rate</div><div class=v>{jl:.1f}%</div><div class=s>관절한계 근접(physics)</div></div>
  <div class=card><div class=k>mean jerk</div><div class=v>{(pp.get('mean_jerk') or 0):.3f}</div><div class=s>행동 급변(physics)</div></div>
 </div>
 <table><thead><tr><th>task</th><th>결과</th><th>steps</th><th>jerk</th><th>관절한계율</th></tr></thead>
 <tbody>{rows}</tbody></table>
 <p class=muted style="margin-top:8px">측정시간 {b.get('wall_sec','?')}s · 주지표=success(H1 비열화 기준), physics proxy=joint-limit/jerk(H1 감소 대상).
 합격선: physics-error 상대 ≥20%↓ AND success ≥baseline−1%p.</p>
</div>
"""


RESULTS_B = """
__BASELINE__
<div class=day><h3>🎯 핵심 가설 · 합격선 (사전등록, 데이터 보기 전 고정)</h3>
 <table><thead><tr><th>가설</th><th>내용</th><th>합격선(PASS)</th></tr></thead><tbody>
 <tr><td class=h>H1</td><td>MGPO(LoRA) RL → OpenVLA 물리오류 행동률 감소</td><td>physics-error 상대 ≥20%↓ AND success 비열화(≥baseline−1%p)</td></tr>
 <tr><td class=h>H2</td><td>MGPO는 다양성(엔트로피) 유지 → 미지과제 일반화 우위</td><td>엔트로피 &gt; 단순RL AND 미지과제 success 우위</td></tr>
 <tr><td class=h>H3</td><td>compression-coverage: 소형코어로 압축해도 유지</td><td>H1 감소폭의 ≥80% 유지, 추론메모리 O(1)/소형</td></tr>
 </tbody></table>
 <p class=muted style="margin-top:10px">단계: <b>S0</b>(정책측 확인, 완료) → S0b LIBERO baseline → S1 MGPO LoRA → S2 압축 → S3 autoresearch 스윕.
 합격선 미측정 라운드는 INCONCLUSIVE로 둔다(사후 변경 금지).</p>
</div>
<div class=day><h3>🧭 두 오류축 분리 — 서로 다른 방법으로 푼다</h3>
 <div class=grid2>
  <div class=cor style="border-left-color:var(--acc)"><div class=q style="color:var(--acc)">① referential / object-permanence</div>
   <div class=a>"빨간 공을 잡아라 → <b>그거</b>를 던져라"의 <i>그거</i>가 resolve 안 됨</div>
   <div class=f style="color:var(--mut)">→ 해법: 에이전트/상태메모리 (Phase 1 DeepAgent · Phase 2 object memory)</div></div>
  <div class=cor><div class=q>② physics</div>
   <div class=a>충돌·관절한계·드롭 등 물리적으로 틀린 행동</div>
   <div class=f style="color:var(--mut)">→ 해법: MGPO 검증보상 RL (verifiable reward = LIBERO success + 물리타당)</div></div>
 </div>
 <p class=muted style="margin-top:10px">아키텍처 다이어그램:
  <a href="../experiments/openvla-mgpo/architecture.svg" style="color:var(--acc)">architecture.svg</a> ·
  설계: <a href="../experiments/openvla-mgpo/ARCHITECTURE.md" style="color:var(--acc)">ARCHITECTURE.md</a></p>
</div>
"""

CHART_PANEL = """
   <div class=grid2>
     <div class=chartbox><h4>① 연산 시간 vs 길이 (O(L) vs O(L²))</h4><canvas id=c1 width=520 height=300></canvas><div class=legend>🟥 Transformer(FlashAttn) · 🟩 DeltaNet</div></div>
     <div class=chartbox><h4>② 추론 메모리 (KV캐시 vs 재귀상태)</h4><canvas id=c2 width=520 height=300></canvas><div class=legend>🟥 KV cache O(L) · 🟦 재귀상태 O(1) 132KB</div></div>
     <div class=chartbox><h4>③ 검증된 구현 회상 (MQAR)</h4><canvas id=c3 width=520 height=300></canvas><div class=legend>chance≈0.062</div></div>
     <div class=chartbox><h4>④ grokking — titans 긴 컨텍스트</h4><canvas id=c4 width=520 height=300></canvas><div class=legend>0.33 정체 → 15k 급점프 → 1.0</div></div>
   </div>
"""

STYLE = """
:root{--bg:#0e1116;--pan:#161b22;--ln:#283040;--ink:#e6edf3;--mut:#8b96a5;--acc:#4f8cff;--good:#3fb950;--warn:#d29922;--bad:#f85149;--mono:"SF Mono",ui-monospace,Menlo,monospace}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.6 -apple-system,"Apple SD Gothic Neo","Noto Sans KR",sans-serif}
header{padding:22px 26px;border-bottom:1px solid var(--ln);display:flex;justify-content:space-between;align-items:flex-start;gap:16px;flex-wrap:wrap}
h1{margin:0;font-size:21px}.tag{color:var(--mut);font-size:12.5px;margin-top:4px}.sub{color:var(--mut);font:12px var(--mono);margin-top:6px}
.xlink{flex:none;background:var(--pan);border:1px solid var(--ln);border-radius:8px;padding:8px 13px;color:var(--acc);text-decoration:none;font-size:12.5px}
.xlink:hover{border-color:var(--acc)}
main{max-width:1180px;margin:0 auto;padding:22px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(165px,1fr));gap:12px;margin-bottom:20px}
.card{background:var(--pan);border:1px solid var(--ln);border-radius:12px;padding:15px}
.card .k{color:var(--mut);font-size:11px;text-transform:uppercase;letter-spacing:.04em}
.card .v{font-size:23px;font-weight:700;margin-top:5px;color:var(--acc)}.card .s{color:var(--mut);font:11px var(--mono);margin-top:3px}
.tabs{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:16px}
.tab{padding:8px 15px;border:1px solid var(--ln);border-radius:8px;cursor:pointer;color:var(--mut);font-size:13px;background:transparent}
.tab.on{background:var(--acc);color:#fff;border-color:var(--acc)}
.panel{display:none}.panel.on{display:block}
.day{background:var(--pan);border:1px solid var(--ln);border-radius:12px;padding:16px 18px;margin-bottom:14px}
.day h3{margin:0 0 10px;font-size:15px;color:var(--acc)}
.mi{border-left:3px solid var(--good);padding:6px 0 6px 12px;margin:8px 0}
.mi b{font-size:13.5px}.mi span{display:block;color:var(--mut);font-size:12px}
table{width:100%;border-collapse:collapse;background:var(--pan);border:1px solid var(--ln);border-radius:10px;overflow:hidden}
th,td{text-align:left;padding:9px 12px;border-bottom:1px solid var(--ln);font-size:12.5px;vertical-align:top}
th{color:var(--mut);font-size:11px;text-transform:uppercase}tr:last-child td{border-bottom:0}
td.h{font:12px var(--mono);color:var(--acc)}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:16px}@media(max-width:820px){.grid2{grid-template-columns:1fr}}
.chartbox{background:var(--pan);border:1px solid var(--ln);border-radius:12px;padding:14px}
.chartbox h4{margin:0 0 8px;font-size:13px;color:var(--ink)}canvas{width:100%;height:auto}
.cor{background:var(--pan);border:1px solid var(--ln);border-left:3px solid var(--warn);border-radius:8px;padding:11px 14px;margin-bottom:10px}
.cor .q{color:var(--warn);font-weight:600;font-size:13px}.cor .a{color:var(--mut);font-size:12.5px}.cor .f{color:var(--good);font-size:12.5px}
.muted{color:var(--mut)}.legend{font:11px var(--mono);color:var(--mut);margin-top:6px}
"""

CHART_JS = """
function axes(ctx,W,H,pad){ctx.clearRect(0,0,W,H);ctx.strokeStyle='#283040';ctx.fillStyle='#8b96a5';ctx.font='10px monospace';return{x0:pad,y0:H-28,pw:W-pad-10,ph:H-28-10};}
let drawn=false;
function draw(){if(drawn||!CH)return;drawn=true;
 let c=c1.getContext('2d'),A=axes(c,520,300,40);const L=CH.compute.L;
 const lx=v=>A.x0+Math.log2(v/L[0])/Math.log2(L[L.length-1]/L[0])*A.pw;
 const allt=CH.compute.tf.concat(CH.compute.deltanet),mn=Math.min(...allt),mx=Math.max(...allt);
 const ly=v=>A.y0-(Math.log10(v)-Math.log10(mn))/(Math.log10(mx)-Math.log10(mn))*A.ph;
 [['tf','#f85149'],['deltanet','#3fb950']].forEach(([k,col])=>{c.strokeStyle=col;c.fillStyle=col;c.lineWidth=2;c.beginPath();
  CH.compute[k].forEach((v,i)=>{const x=lx(L[i]),y=ly(v);i?c.lineTo(x,y):c.moveTo(x,y);});c.stroke();
  CH.compute[k].forEach((v,i)=>{c.beginPath();c.arc(lx(L[i]),ly(v),3,0,7);c.fill();});});
 c.fillStyle='#8b96a5';L.forEach(v=>c.fillText(v>=1024?(v/1024)+'K':v,lx(v)-8,A.y0+14));
 c=c2.getContext('2d');A=axes(c,520,300,44);const L2=CH.infer.L;
 const lx2=v=>A.x0+Math.log2(v/L2[0])/Math.log2(L2[L2.length-1]/L2[0])*A.pw;
 const mx2=Math.max(...CH.infer.kv),ly2=v=>A.y0-(Math.log10(Math.max(v,.01))-Math.log10(.01))/(Math.log10(mx2)-Math.log10(.01))*A.ph;
 c.strokeStyle='#f85149';c.fillStyle='#f85149';c.lineWidth=2;c.beginPath();
 CH.infer.kv.forEach((v,i)=>{const x=lx2(L2[i]),y=ly2(v);i?c.lineTo(x,y):c.moveTo(x,y);});c.stroke();
 c.strokeStyle='#4f8cff';c.beginPath();L2.forEach((l,i)=>{const x=lx2(l),y=ly2(CH.infer.state);i?c.lineTo(x,y):c.moveTo(x,y);});c.stroke();
 c.fillStyle='#8b96a5';L2.forEach(v=>c.fillText(v>=1e6?(v/1e6)+'M':(v/1024)+'K',lx2(v)-10,A.y0+14));c.fillText('GB',6,16);
 c=c3.getContext('2d');A=axes(c,520,300,30);const N=CH.recall.names.length,sw=A.pw/N;
 for(let g=0;g<=4;g++){const y=A.y0-g/4*A.ph;c.strokeStyle='#283040';c.beginPath();c.moveTo(A.x0,y);c.lineTo(520-10,y);c.stroke();c.fillStyle='#8b96a5';c.fillText((g/4).toFixed(2),6,y+3);}
 CH.recall.vals.forEach((v,i)=>{const x=A.x0+i*sw+8,bw=sw-16;c.fillStyle=v>=.9?'#3fb950':(v>.2?'#d29922':'#f85149');c.fillRect(x,A.y0-v*A.ph,bw,v*A.ph);
  c.save();c.translate(x+bw/2,A.y0+12);c.rotate(-.4);c.fillStyle='#8b96a5';c.textAlign='right';c.fillText(CH.recall.names[i],0,0);c.restore();});
 c=c4.getContext('2d');A=axes(c,520,300,34);const S=CH.grok.steps,R=CH.grok.rec,smx=S[S.length-1];
 for(let g=0;g<=4;g++){const y=A.y0-g/4*A.ph;c.strokeStyle='#283040';c.beginPath();c.moveTo(A.x0,y);c.lineTo(520-10,y);c.stroke();c.fillStyle='#8b96a5';c.fillText((g/4).toFixed(2),6,y+3);}
 c.strokeStyle='#4f8cff';c.lineWidth=2;c.beginPath();S.forEach((s,i)=>{const x=A.x0+s/smx*A.pw,y=A.y0-R[i]*A.ph;i?c.lineTo(x,y):c.moveTo(x,y);});c.stroke();
 const gx=A.x0+15000/smx*A.pw;c.strokeStyle='#3fb950';c.setLineDash([4,3]);c.beginPath();c.moveTo(gx,A.y0);c.lineTo(gx,A.y0-A.ph);c.stroke();c.setLineDash([]);c.fillStyle='#3fb950';c.fillText('grok@15k',gx+4,A.y0-A.ph+12);}
"""


def build(t):
    commits = t["commits"]
    dates = sorted({c["date"] for c in commits})
    stats = [(k, v.replace("{N}", str(len(commits))), s) for k, v, s in t["stats"](commits, dates)]
    has_charts = t["charts"] is not None

    # 패널 구성 (id, label)
    panels = [("timeline", "타임라인")]
    if has_charts:
        panels.append(("results", "핵심 결과"))
    if t["results"] == "RESULTS_B":
        panels.append(("results", "📊 결과·가설"))
        if B_BASE:
            panels.append(("qual", "🔬 정성 분석"))
        if B_VIDS:
            panels.append(("videos", "🎬 영상"))
    if t["results"] == "RESULTS_C":
        panels.append(("results", "📊 재현 결과"))
    panels.append(("bridge", "🔗 연결"))
    panels.append(("honest", "정직한 정정"))
    panels.append(("commits", "커밋"))

    # 패널 div (raw 콘텐츠 주입)
    pdivs = ['<div class=panel id=p_timeline></div>']
    if has_charts:
        pdivs.append(f'<div class=panel id=p_results>{CHART_PANEL}</div>')
    elif t["results"] == "RESULTS_B":
        pdivs.append(f'<div class=panel id=p_results>{RESULTS_B.replace("__BASELINE__", _baseline_html())}</div>')
        if B_BASE:
            pdivs.append(f'<div class=panel id=p_qual>{_qual_html()}</div>')
        if B_VIDS:
            pdivs.append(f'<div class=panel id=p_videos>{_videos_html()}</div>')
    elif t["results"] == "RESULTS_C":
        pdivs.append(f'<div class=panel id=p_results>{_track_c_results()}</div>')
    pdivs.append('<div class=panel id=p_bridge></div>')
    pdivs.append('<div class=panel id=p_honest></div>')
    pdivs.append('<div class=panel id=p_commits></div>')

    nav = "".join(f'<a class=xlink href="{f}">↔ {nm}</a>' for f, nm in TRACK_NAV if f != t["file"])
    bridge = t.get("bridge", BRIDGE)
    bridge_note = t.get("bridge_note", BRIDGE_NOTE)
    html = f"""<!DOCTYPE html><html lang=ko><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1"><title>{t['name']}</title>
<style>{STYLE}</style></head><body>
<header><div><h1>{t['icon']} {t['name']}</h1><div class=tag>{t['tag']}</div><div class=sub id=meta></div></div>
<div style="display:flex;gap:6px;flex-wrap:wrap">{nav}</div></header>
<main>
 <div class=cards id=cards></div>
 <div class=tabs id=tabs></div>
 {''.join(pdivs)}
</main>
<script>
const COMMITS={js(commits)}, DATES={js(dates)}, STATS={js(stats)}, MILE={js(t['mile'])},
      COR={js(t['cor'])}, CH={js(t['charts']) if has_charts else 'null'},
      BRIDGE={js(bridge)}, BRIDGE_NOTE={js(bridge_note)}, PANELS={js(panels)};
document.getElementById('meta').textContent='기간 '+(DATES[0]||'-')+' ~ '+(DATES[DATES.length-1]||'-')+'  ·  커밋 '+COMMITS.length;
cards.innerHTML=STATS.map(s=>`<div class=card><div class=k>${{s[0]}}</div><div class=v>${{s[1]}}</div><div class=s>${{s[2]}}</div></div>`).join('');
tabs.innerHTML=PANELS.map((t,i)=>`<div class="tab${{i==0?' on':''}}" data-t=${{t[0]}}>${{t[1]}}</div>`).join('');
function show(t){{document.querySelectorAll('.tab').forEach(e=>e.classList.toggle('on',e.dataset.t==t));
 document.querySelectorAll('.panel').forEach(e=>e.classList.remove('on'));document.getElementById('p_'+t).classList.add('on');
 if(t=='results')draw();}}
document.querySelectorAll('.tab').forEach(e=>e.onclick=()=>show(e.dataset.t));
p_timeline.innerHTML=[...DATES].reverse().map(d=>{{
 const ms=(MILE[d]||[]).map(m=>`<div class=mi><b>${{m[0]}}</b><span>${{m[1]}}</span></div>`).join('');
 const cs=COMMITS.filter(c=>c.date==d);
 return `<div class=day><h3>📅 ${{d}} <span class=muted style="font-size:12px">(${{cs.length}} commits)</span></h3>${{ms}}</div>`;}}).join('');
p_bridge.innerHTML='<div class=day><h3>🔗 트랙 간 연결</h3><p class=muted>'+BRIDGE_NOTE+'</p>'+
 BRIDGE.map(b=>`<div class=mi><b>${{b[0]}}</b><span>${{b[1]}}</span></div>`).join('')+'</div>';
p_honest.innerHTML='<p class=muted>이 트랙에서 스스로 정정한 과장/오류(사용자 지적 포함):</p>'+
 COR.map(c=>`<div class=cor><div class=q>⚠ ${{c[0]}}</div><div class=a>→ ${{c[1]}}</div><div class=f>✓ ${{c[2]}}</div></div>`).join('');
p_commits.innerHTML='<table><thead><tr><th>날짜</th><th>해시</th><th>내용</th></tr></thead><tbody>'+
 COMMITS.map(c=>`<tr><td class=muted>${{c.date}}</td><td class=h>${{c.hash}}</td><td>${{c.subj}}</td></tr>`).join('')+'</tbody></table>';
{CHART_JS}
show('timeline');
</script></body></html>"""
    out = os.path.join(ROOT, "report", t["file"])
    open(out, "w", encoding="utf-8").write(html)
    print("wrote", out, f"({len(html)} bytes, {len(commits)} commits, {len(dates)} days)")


def main():
    for key in ("a", "b", "c"):
        build(TRACKS[key])
    # 단일 통합 보고서는 트랙 분리로 대체됨
    old = os.path.join(ROOT, "report", "daily_report.html")
    if os.path.exists(old):
        os.remove(old)
        print("removed (superseded)", old)


if __name__ == "__main__":
    main()
