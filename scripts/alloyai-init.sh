#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: alloyai-init.sh [--install-dir=/usr/alloyai] [--source-dir=/path/to/source] [--extras=all] [--python=python3]

Creates a venv under the install dir and installs the package from source.
USAGE
}

INSTALL_DIR="/usr/alloyai"
SOURCE_DIR=""
EXTRAS="all"
PYTHON_BIN="python3"

for arg in "$@"; do
  case "$arg" in
    --install-dir=*) INSTALL_DIR="${arg#*=}" ;;
    --source-dir=*) SOURCE_DIR="${arg#*=}" ;;
    --extras=*) EXTRAS="${arg#*=}" ;;
    --python=*) PYTHON_BIN="${arg#*=}" ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $arg" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "$SOURCE_DIR" ]]; then
  SOURCE_DIR="$INSTALL_DIR/src"
fi

if [[ ! -d "$SOURCE_DIR" ]]; then
  echo "Source directory not found: $SOURCE_DIR" >&2
  exit 1
fi

VENV_DIR="$INSTALL_DIR/venv"

mkdir -p "$INSTALL_DIR"
if [[ ! -d "$VENV_DIR" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

"$VENV_DIR/bin/pip" install --upgrade pip

if [[ -n "$EXTRAS" && "$EXTRAS" != "none" ]]; then
  "$VENV_DIR/bin/pip" install "$SOURCE_DIR[$EXTRAS]"
else
  "$VENV_DIR/bin/pip" install "$SOURCE_DIR"
fi

echo "Venv ready at: $VENV_DIR"
