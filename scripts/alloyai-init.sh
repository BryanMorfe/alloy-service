#!/usr/bin/env bash
set -euo pipefail

LOG_PREFIX="[ALLOYAI]"
LOG_LEVEL="info"

_level_value() {
  case "$1" in
    debug) echo 10 ;;
    info) echo 20 ;;
    warn|warning) echo 30 ;;
    error) echo 40 ;;
    *) echo 20 ;;
  esac
}

_log() {
  local level="$1"
  shift
  local current_level="$(_level_value "$LOG_LEVEL")"
  local msg_level="$(_level_value "$level")"
  if [[ "$msg_level" -lt "$current_level" ]]; then
    return
  fi
  printf '%s %s %s %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$LOG_PREFIX" "${level^^}" "$*" >&2
}

usage() {
  cat <<'USAGE'
Usage: alloyai-init.sh [--install-dir=/usr/local/alloyai] [--source-dir=/path/to/source] [--extras=all] [--python=python3]

Creates a venv under the install dir and installs the package from source.
USAGE
}

INSTALL_DIR="/usr/local/alloyai"
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
  _log error "Source directory not found: $SOURCE_DIR"
  exit 1
fi

VENV_DIR="$INSTALL_DIR/venv"

mkdir -p "$INSTALL_DIR"
if [[ ! -d "$VENV_DIR" ]]; then
  _log info "Creating venv at $VENV_DIR"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

_log info "Upgrading pip in venv"
"$VENV_DIR/bin/pip" install --upgrade pip

if [[ -n "$EXTRAS" && "$EXTRAS" != "none" ]]; then
  _log info "Installing package with extras: $EXTRAS"
  "$VENV_DIR/bin/pip" install "$SOURCE_DIR[$EXTRAS]"
else
  _log info "Installing package"
  "$VENV_DIR/bin/pip" install "$SOURCE_DIR"
fi

_log info "Venv ready at: $VENV_DIR"
