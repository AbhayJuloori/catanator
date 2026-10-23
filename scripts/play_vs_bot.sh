#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${CATAN_PYTHON:-${PROJECT_ROOT}/.venv/bin/python}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python environment not found at ${PYTHON_BIN}" >&2
  echo "Set CATAN_PYTHON to the project's Python 3.11 executable." >&2
  exit 1
fi

if ! PYTHONPATH="${PROJECT_ROOT}" "${PYTHON_BIN}" -c \
  'from catan_ranked_fast import Game, topology_json; assert hasattr(Game(0), "state_json"); topology_json()' >/dev/null 2>&1; then
  echo "The play-UI fast engine is not installed in ${PYTHON_BIN}." >&2
  echo "From this worktree, run once:" >&2
  echo "  .venv/bin/maturin develop --release --manifest-path fast_engine/Cargo.toml" >&2
  exit 1
fi

cd "${PROJECT_ROOT}"
PLAY_PORT="${CATAN_PLAY_PORT:-8765}"
echo "Opening Catanator at http://localhost:${PLAY_PORT}"
exec env PYTHONPATH=. "${PYTHON_BIN}" -m catan_ranked.play_ui
