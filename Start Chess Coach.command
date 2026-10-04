#!/bin/zsh
# Double-click this file, or run it from Codex with camera access.
cd -- "$(dirname -- "$0")" || exit 1
if [[ ! -x .venv/bin/python ]]; then
  print 'First run: bash scripts/setup.sh'
  exit 1
fi
export PYTHONDONTWRITEBYTECODE=1
exec .venv/bin/python main.py --top-color black --board-rotation cw "$@"
