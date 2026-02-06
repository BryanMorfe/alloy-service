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
Usage: alloyai-install.sh [--install-dir=/usr/local/alloyai] [--force]

Copies source into the install dir and installs helper scripts.
USAGE
}

INSTALL_DIR="/usr/local/alloyai"
FORCE=0

for arg in "$@"; do
  case "$arg" in
    --install-dir=*) INSTALL_DIR="${arg#*=}" ;;
    --force) FORCE=1 ;;
    -h|--help) usage; exit 0 ;;
    *) _log error "Unknown argument: $arg"; usage; exit 1 ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ ! -d "$SOURCE_DIR/alloyai" ]]; then
  _log error "Expected alloyai package in $SOURCE_DIR"
  exit 1
fi

SRC_DEST="$INSTALL_DIR/src"
BIN_DEST="$INSTALL_DIR/bin"

if [[ -d "$SRC_DEST" && $FORCE -ne 1 ]]; then
  _log error "Install directory already exists: $SRC_DEST (use --force to overwrite)"
  exit 1
fi

mkdir -p "$SRC_DEST" "$BIN_DEST"

if [[ $FORCE -eq 1 ]]; then
  _log warn "Overwriting existing source directory"
  rm -rf "$SRC_DEST/alloyai"
fi

_log info "Copying source to $SRC_DEST"
cp -a "$SOURCE_DIR/alloyai" "$SRC_DEST/"
for file in pyproject.toml README.md LICENSE AGENTS.md; do
  if [[ -f "$SOURCE_DIR/$file" ]]; then
    cp -a "$SOURCE_DIR/$file" "$SRC_DEST/"
  fi
done

_log info "Installing scripts to $BIN_DEST"
cp -a "$SCRIPT_DIR/alloyai-init.sh" "$BIN_DEST/"
cp -a "$SCRIPT_DIR/alloyai-start.sh" "$BIN_DEST/"
cp -a "$SCRIPT_DIR/alloyai.service" "$BIN_DEST/"

chmod +x "$BIN_DEST/alloyai-init.sh" "$BIN_DEST/alloyai-start.sh"

CONFIG_DIR="/etc/alloyai"
CONFIG_FILE="$CONFIG_DIR/alloyai.conf"

if mkdir -p "$CONFIG_DIR" 2>/dev/null; then
  if [[ ! -f "$CONFIG_FILE" ]]; then
    cat > "$CONFIG_FILE" <<'CONF'
# AlloyAI server config
host = 0.0.0.0
port = 8000
log_level = info
log_prefix = [ALLOYAI]
# log_format = %(asctime)s [ALLOYAI] %(levelname)s %(name)s: %(message)s
# extra_args = --log-level info
CONF
  fi
else
  _log warn "Could not create $CONFIG_DIR (insufficient permissions)."
  _log warn "Create $CONFIG_FILE manually if needed."
fi

_log info "Installed source to: $SRC_DEST"
_log info "Scripts in: $BIN_DEST"
_log info "Config: $CONFIG_FILE"
