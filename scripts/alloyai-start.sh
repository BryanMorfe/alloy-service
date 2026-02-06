#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: alloyai-start.sh [--install-dir=/usr/alloyai] [--config=/etc/alloyai/alloyai.conf]

Runs AlloyAI using the venv and config file.
USAGE
}

INSTALL_DIR="/usr/alloyai"
CONFIG_FILE="/etc/alloyai/alloyai.conf"

for arg in "$@"; do
  case "$arg" in
    --install-dir=*) INSTALL_DIR="${arg#*=}" ;;
    --config=*) CONFIG_FILE="${arg#*=}" ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $arg" >&2; usage; exit 1 ;;
  esac
done

VENV_BIN="$INSTALL_DIR/venv/bin"
ALLOY_BIN="$VENV_BIN/alloy"

if [[ ! -x "$ALLOY_BIN" ]]; then
  echo "Alloy executable not found: $ALLOY_BIN" >&2
  exit 1
fi

read_conf() {
  local key="$1"
  if [[ ! -f "$CONFIG_FILE" ]]; then
    return 0
  fi
  awk -F'=' -v key="$key" '
    /^[[:space:]]*#/ {next}
    NF < 2 {next}
    $1 ~ "^[[:space:]]*" key "[[:space:]]*$" {
      val=$2
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", val)
      print val
      exit
    }
  ' "$CONFIG_FILE"
}

HOST="${ALLOYAI_HOST:-$(read_conf host)}"
PORT="${ALLOYAI_PORT:-$(read_conf port)}"
EXTRA_ARGS="$(read_conf extra_args)"

if [[ -z "$HOST" ]]; then
  HOST="0.0.0.0"
fi
if [[ -z "$PORT" ]]; then
  PORT="8000"
fi

exec "$ALLOY_BIN" serve --host "$HOST" --port "$PORT" ${EXTRA_ARGS:-}
