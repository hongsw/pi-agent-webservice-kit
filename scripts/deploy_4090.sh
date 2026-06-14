#!/usr/bin/env bash
# AutoResearch 노드를 4090 리눅스 머신에 배포하고 스윕을 실행한다.
# 사용: scripts/deploy_4090.sh [run|web|stop]
set -euo pipefail

HOST="${AR_HOST:-martin@linux-builder}"
DEST="${AR_DEST:-~/dev/growing-memory-node}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"
CMD="${1:-run}"

echo "[deploy] rsync $HERE -> $HOST:$DEST"
rsync -az --delete \
  --exclude '.git' --exclude '__pycache__' --exclude 'runs' \
  --exclude '.ratchet_snapshots' --exclude '.runs' --exclude '*.pyc' \
  -e "ssh -o BatchMode=yes" \
  "$HERE/"  "$HOST:$DEST/"

case "$CMD" in
  run)
    echo "[deploy] 스윕 실행(실물 growing-memory 있으면 GROWING_MEMORY_HOME 설정)"
    ssh -o BatchMode=yes "$HOST" "cd $DEST/tutorial/autoresearch && \
      python3 run.py run --config config/run_example.yaml --leaderboard runs/leaderboard.jsonl"
    ;;
  web)
    echo "[deploy] Web 대시보드 기동(포트 8765). Ctrl-C로 종료."
    ssh -o BatchMode=yes -t "$HOST" "cd $DEST && PORT=8765 python3 web/server.py"
    ;;
  stop)
    ssh -o BatchMode=yes "$HOST" "pkill -f 'web/server.py' || true; echo stopped"
    ;;
  *)
    echo "usage: $0 [run|web|stop]"; exit 1;;
esac
