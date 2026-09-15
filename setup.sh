#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Error: $PYTHON_BIN was not found."
  echo "Install Python 3.10+ and run this script again."
  exit 1
fi

"$PYTHON_BIN" - <<'PY'
import sys
if sys.version_info < (3, 10):
    raise SystemExit(
        f"Python 3.10+ is required. Current version: {sys.version.split()[0]}"
    )
print(f"Using Python {sys.version.split()[0]}")
PY

if [[ ! -d "$VENV_DIR" ]]; then
  echo "Creating virtual environment: $VENV_DIR"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
else
  echo "Virtual environment already exists: $VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt

if [[ ! -f ".env" ]]; then
  cp .env.example .env
  echo "Created .env from .env.example"
else
  echo ".env already exists; leaving it unchanged."
fi

python - <<'PY'
import torch, transformers, fastapi, peft, datasets
print("Environment check passed.")
print(f"PyTorch      : {torch.__version__}")
print(f"Transformers : {transformers.__version__}")
print(f"FastAPI      : {fastapi.__version__}")
print(f"CUDA visible : {torch.cuda.is_available()}")
PY

echo
echo "Setup complete."
echo "Start the API/dashboard with:"
echo "  ./launch.sh"
echo
echo "Open the latest training notebook with:"
echo "  ./launch.sh notebook"
