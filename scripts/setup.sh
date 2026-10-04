#!/bin/bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$project_dir"
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
if command -v brew >/dev/null 2>&1; then
  HOMEBREW_NO_AUTO_UPDATE=1 brew install stockfish
else
  printf '%s\n' 'Install Stockfish and use --engine-path /path/to/stockfish.'
fi
printf '%s\n' 'Ready: run ./Start\ Chess\ Coach.command' \
  'Optional local voice: bash scripts/setup_voice.sh'
