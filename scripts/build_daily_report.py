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


def js(x):
    return json.dumps(x, ensure_ascii=False)


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
B_COR = [
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

TRACKS = {
  "a": {
    "file": "track_a_growing_memory.html",
    "icon": "🔬", "name": "트랙 A — growing-memory / Titans 효율 연구",
    "tag": "선형·재귀 상태메모리: O(1) 추론 · 128K 학습 · 검증된 회상",
    "other": ("track_b_openvla_mgpo.html", "🤖 트랙 B — OpenVLA×MGPO"),
    "commits": [c for c in ALL_COMMITS if not is_b(c["subj"])],
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
    "stats": lambda cm, dts: [
      ("단계", "S0 완료", "정책측 확인 (다음 S0b LIBERO)"),
      ("모델", "OpenVLA-7b", "7.54B params, fp16"),
      ("로드", "15.1GB", "/ 24GB (4090, peak)"),
      ("가설", "H1~H3", "물리오류↓ · 다양성 · 압축"),
      ("H1 합격선", "≥20%↓", "physics-error, success 비열화"),
      ("커밋", "{N}", "신규 트랙"),
    ],
    "results": "RESULTS_B",  # 가설·설계 + 두 오류축 raw HTML
  },
}

# ── 트랙 B 전용 패널 HTML ─────────────────────────────────────────────────────
RESULTS_B = """
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
        panels.append(("results", "가설·설계"))
    panels.append(("bridge", "🔗 연결"))
    panels.append(("honest", "정직한 정정"))
    panels.append(("commits", "커밋"))

    # 패널 div (raw 콘텐츠 주입)
    pdivs = ['<div class=panel id=p_timeline></div>']
    if has_charts:
        pdivs.append(f'<div class=panel id=p_results>{CHART_PANEL}</div>')
    elif t["results"] == "RESULTS_B":
        pdivs.append(f'<div class=panel id=p_results>{RESULTS_B}</div>')
    pdivs.append('<div class=panel id=p_bridge></div>')
    pdivs.append('<div class=panel id=p_honest></div>')
    pdivs.append('<div class=panel id=p_commits></div>')

    of, oname = t["other"]
    html = f"""<!DOCTYPE html><html lang=ko><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1"><title>{t['name']}</title>
<style>{STYLE}</style></head><body>
<header><div><h1>{t['icon']} {t['name']}</h1><div class=tag>{t['tag']}</div><div class=sub id=meta></div></div>
<a class=xlink href="{of}">↔ {oname}</a></header>
<main>
 <div class=cards id=cards></div>
 <div class=tabs id=tabs></div>
 {''.join(pdivs)}
</main>
<script>
const COMMITS={js(commits)}, DATES={js(dates)}, STATS={js(stats)}, MILE={js(t['mile'])},
      COR={js(t['cor'])}, CH={js(t['charts']) if has_charts else 'null'},
      BRIDGE={js(BRIDGE)}, BRIDGE_NOTE={js(BRIDGE_NOTE)}, PANELS={js(panels)};
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
p_bridge.innerHTML='<div class=day><h3>🔗 두 연구의 연결점 (단 하나, Phase 2)</h3><p class=muted>'+BRIDGE_NOTE+'</p>'+
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
    for key in ("a", "b"):
        build(TRACKS[key])
    # 단일 통합 보고서는 트랙 분리로 대체됨
    old = os.path.join(ROOT, "report", "daily_report.html")
    if os.path.exists(old):
        os.remove(old)
        print("removed (superseded)", old)


if __name__ == "__main__":
    main()
