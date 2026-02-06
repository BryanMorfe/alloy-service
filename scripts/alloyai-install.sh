#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: alloyai-install.sh [--install-dir=/usr/alloyai] [--force]

Copies source into the install dir and installs helper scripts.
USAGE
}

INSTALL_DIR="/usr/alloyai"
FORCE=0

for arg in "$@"; do
  case "$arg" in
    --install-dir=*) INSTALL_DIR="${arg#*=}" ;;
    --force) FORCE=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $arg" >&2; usage; exit 1 ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ ! -d "$SOURCE_DIR/alloyai" ]]; then
  echo "Expected alloyai package in $SOURCE_DIR" >&2
  exit 1
fi

SRC_DEST="$INSTALL_DIR/src"
BIN_DEST="$INSTALL_DIR/bin"

if [[ -d "$SRC_DEST" && $FORCE -ne 1 ]]; then
  echo "Install directory already exists: $SRC_DEST (use --force to overwrite)" >&2
  exit 1
fi

mkdir -p "$SRC_DEST" "$BIN_DEST"

if [[ $FORCE -eq 1 ]]; then
  rm -rf "$SRC_DEST/alloyai"
fi

cp -a "$SOURCE_DIR/alloyai" "$SRC_DEST/"
for file in pyproject.toml README.md LICENSE AGENTS.md; do
  if [[ -f "$SOURCE_DIR/$file" ]]; then
    cp -a "$SOURCE_DIR/$file" "$SRC_DEST/"
  fi
done

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
# extra_args = --log-level info
CONF
  fi
else
  echo "Could not create $CONFIG_DIR (insufficient permissions)."
  echo "Create $CONFIG_FILE manually if needed."
fi

echo "Installed source to: $SRC_DEST"
echo "Scripts in: $BIN_DEST"
echo "Config: $CONFIG_FILE"
