#!/usr/bin/env bash
set -e

PYTHON_EXE="${PYTHON_EXE:-python}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_PATH="$PROJECT_ROOT/.venv"

if [ ! -d "$VENV_PATH" ]; then
  echo "Creating virtual environment in $VENV_PATH"
  "$PYTHON_EXE" -m venv "$VENV_PATH"
fi

VENV_PYTHON="$VENV_PATH/bin/python"
REQUIREMENTS_FILE="$PROJECT_ROOT/requirements.txt"

echo "Upgrading pip in virtual environment..."
"$VENV_PYTHON" -m pip install --upgrade pip

echo "Installing dependencies from $REQUIREMENTS_FILE ..."
"$VENV_PYTHON" -m pip install -r "$REQUIREMENTS_FILE"


