#!/usr/bin/env bash
# md 문서 → 시각화 PDF. pandoc(md→HTML) + Chrome 헤드리스(HTML→PDF) + matplotlib 차트.
# 사용: scripts/build_pdf.sh
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CSS="$ROOT/scripts/pdf_style.css"
OUT="$ROOT/report/pdf"; FIG="$ROOT/report/figures"
TMP="$(mktemp -d)"
CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
mkdir -p "$OUT"

echo "[1/3] 차트 생성"; python3 "$ROOT/scripts/gen_charts.py"

html2pdf() {  # $1=html $2=pdf
  "$CHROME" --headless=new --disable-gpu --no-pdf-header-footer \
    --virtual-time-budget=8000 --run-all-compositor-stages-before-draw \
    --print-to-pdf="$2" "file://$1" >/dev/null 2>&1 || \
  "$CHROME" --headless --disable-gpu --print-to-pdf="$2" "file://$1" >/dev/null 2>&1
}
md2pdf() {    # $1=md $2=pdf [extra resource path]
  local h="$TMP/$(basename "$1").html"
  pandoc "$1" -f gfm -t html5 --standalone --embed-resources \
    --metadata title="$(basename "$1" .md)" --css "$CSS" \
    --resource-path="$ROOT:$ROOT/report:$ROOT/tutorial/autoresearch" -o "$h" 2>/dev/null
  html2pdf "$h" "$2"; echo "  → $(basename "$2")"
}

echo "[2/3] 통합 보고서(REPORT_full.pdf)"
MASTER="$TMP/master.md"
{
  cat "$ROOT/report/REPORT.md"
  echo; echo "# 그림 (실측 시각화)"; echo
  echo "## 연산 시간 vs 길이"; echo "![](figures/fig1_compute_time.png)"; echo
  echo "## 추론 메모리"; echo "![](figures/fig2_inference_memory.png)"; echo
  echo "## 검증된 구현 회상"; echo "![](figures/fig3_recall_by_variant.png)"; echo
  echo "## grokking 곡선"; echo "![](figures/fig4_grokking.png)"; echo
  for d in ROADMAP IMPACT COMPARE3 INFERENCE RECURRENT LONGSEQ GROKKING FLA_VALIDATION TITANS HCSR04 RESULTS; do
    f="$ROOT/report/$d.md"; [ -f "$f" ] || f="$ROOT/tutorial/autoresearch/$d.md"
    [ -f "$f" ] && { echo; cat "$f"; echo; }
  done
} > "$MASTER"
# master는 report/ 기준 상대경로(figures/...) → report/에서 pandoc 실행
( cd "$ROOT/report" && pandoc "$MASTER" -f gfm -t html5 --standalone --embed-resources \
    --metadata title="AutoResearch 노드 — 통합 보고서" --css "$CSS" \
    --resource-path="$ROOT/report:$ROOT" -o "$TMP/master.html" 2>/dev/null )
html2pdf "$TMP/master.html" "$OUT/REPORT_full.pdf"; echo "  → REPORT_full.pdf"

echo "[3/3] 개별 문서 PDF"
md2pdf "$ROOT/report/REPORT.md"   "$OUT/REPORT.pdf"
md2pdf "$ROOT/report/ROADMAP.md"  "$OUT/ROADMAP.pdf"
for d in IMPACT COMPARE3 INFERENCE RECURRENT LONGSEQ GROKKING FLA_VALIDATION TITANS HCSR04 RESULTS; do
  md2pdf "$ROOT/tutorial/autoresearch/$d.md" "$OUT/$d.pdf"
done
rm -rf "$TMP"
echo "완료 → $OUT"; ls -la "$OUT" | awk '{print $5, $NF}' | grep pdf