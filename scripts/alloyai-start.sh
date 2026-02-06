#!/usr/bin/env bash
set -euo pipefail

LOG_PREFIX="[ALLOYAI]"
LOG_LEVEL="info"
LOG_FORMAT=""

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
Usage: alloyai-start.sh [--install-dir=/usr/local/alloyai] [--config=/etc/alloyai/alloyai.conf]

Runs AlloyAI using the venv and config file.
USAGE
}

INSTALL_DIR="/usr/local/alloyai"
CONFIG_FILE="/etc/alloyai/alloyai.conf"

for arg in "$@"; do
  case "$arg" in
    --install-dir=*) INSTALL_DIR="${arg#*=}" ;;
    --config=*) CONFIG_FILE="${arg#*=}" ;;
    -h|--help) usage; exit 0 ;;
    *) _log error "Unknown argument: $arg"; usage; exit 1 ;;
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
      sub(/[[:space:]]*#.*/, "", val)
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", val)
      print val
      exit
    }
  ' "$CONFIG_FILE"
}

HOST="${ALLOYAI_HOST:-$(read_conf host)}"
PORT="${ALLOYAI_PORT:-$(read_conf port)}"
LOG_LEVEL="${ALLOYAI_LOG_LEVEL:-$(read_conf log_level)}"
LOG_PREFIX="${ALLOYAI_LOG_PREFIX:-$(read_conf log_prefix)}"
LOG_FORMAT="${ALLOYAI_LOG_FORMAT:-$(read_conf log_format)}"
EXTRA_ARGS="$(read_conf extra_args)"

if [[ -z "$HOST" ]]; then
  HOST="0.0.0.0"
fi
if [[ -z "$PORT" ]]; then
  PORT="8000"
fi

LOG_ARGS=()
if [[ -n "$LOG_LEVEL" ]]; then
  LOG_ARGS+=(--log-level "$LOG_LEVEL")
fi
if [[ -n "$LOG_PREFIX" ]]; then
  LOG_ARGS+=(--log-prefix "$LOG_PREFIX")
fi
if [[ -n "$LOG_FORMAT" ]]; then
  LOG_ARGS+=(--log-format "$LOG_FORMAT")
fi

_log info "Starting AlloyAI (host=$HOST port=$PORT)"
exec "$ALLOY_BIN" serve --host "$HOST" --port "$PORT" "${LOG_ARGS[@]}" ${EXTRA_ARGS:-}
