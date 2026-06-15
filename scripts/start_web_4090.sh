#!/usr/bin/env bash
# 4090에서 Web UI 기동(네트워크 접속 가능). tmux/nohup로 분리 실행 권장.
cd "$(dirname "$0")/.." || exit 1
export GROWING_MEMORY_HOME="$PWD/packages/growing-memory"
export PORT="${PORT:-8765}"
exec ~/gm_venv/bin/python -u web/server.py
