#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

VENV_DIR="${VENV_DIR:-.venv}"

usage() {
  cat <<'EOF'
Usage:
  ./launch.sh                 Start FastAPI + web dashboard
  ./launch.sh api             Same as above
  ./launch.sh notebook        Open latest v4 Jigsaw notebook
  ./launch.sh path/to/x.ipynb Open a specific notebook
  ./launch.sh --help          Show this help
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

if [[ ! -d "$VENV_DIR" ]]; then
  echo "Virtual environment not found. Run ./setup.sh first."
  exit 1
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

MODE="${1:-api}"

case "$MODE" in
  api)
    HOST="${HOST:-0.0.0.0}"
    PORT="${PORT:-8000}"
    echo "Starting moderation service at http://localhost:${PORT}"
    echo "Swagger docs: http://localhost:${PORT}/docs"
    exec uvicorn app.main:app --reload --host "$HOST" --port "$PORT"
    ;;
  notebook)
    NOTEBOOK="notebooks/bertweet_moderation_model_v4_bertweet_aligned.ipynb"
    ;;
  *)
    NOTEBOOK="$MODE"
    ;;
esac

if [[ ! -f "$NOTEBOOK" ]]; then
  echo "Notebook not found: $NOTEBOOK"
  exit 1
fi

exec jupyter lab "$NOTEBOOK"
