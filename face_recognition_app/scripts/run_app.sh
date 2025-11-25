#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python"

if [ ! -x "$VENV_PYTHON" ]; then
  echo "Virtual environment not found at $VENV_PYTHON"
  echo "Run scripts/install_dependencies.sh first to create .venv and install dependencies."
  exit 1
fi

cd "$PROJECT_ROOT"
"$VENV_PYTHON" -m app.main


