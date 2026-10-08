#!/usr/bin/env bash
set -euo pipefail

# Bootstrap an external noai-watermark checkout for the optional CtrlRegen
# pixel-removal backend.
#
# The upstream project (https://github.com/mertizci/noai-watermark) does not
# ship a LICENSE file, so its code is treated as all-rights-reserved and is
# NOT bundled in this repository. This script clones it locally and installs
# only the dependencies needed by clean_ctrlregen.py.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_DIR="${NOAI_WATERMARK_DIR:-$HOME/noai-watermark}"
DIR=""
# Pinned upstream commit (2026-08-13). Do not point at a moving branch.
REF="b642ae45d20eded52c96d570985eb4e3e427aac8"
PYTHON="${PYTHON:-python3}"

usage() {
  cat <<'EOF'
Usage: setup_ctrlregen.sh [--dir PATH] [--ref REF] [--python PYTHON]

Clones (if needed) mertizci/noai-watermark, creates a venv, and installs the
Python dependencies required by clean_ctrlregen.py (including torch).

Options:
  --dir PATH     checkout directory (default: $NOAI_WATERMARK_DIR or ~/noai-watermark)
  --ref REF      git ref to checkout (default: pinned commit SHA)
  --python PY    Python interpreter used to create the venv (default: python3)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dir)
      DIR="${2:?--dir requires a value}"
      shift 2
      ;;
    --ref)
      REF="${2:?--ref requires a value}"
      shift 2
      ;;
    --python)
      PYTHON="${2:?--python requires a value}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

DIR="${DIR:-$DEFAULT_DIR}"
mkdir -p "$(dirname "$DIR")"
if realpath -m . >/dev/null 2>&1; then  # BSD/macOS realpath has no -m
  DIR="$(realpath -m "$DIR")"
else
  DIR="$(cd "$(dirname "$DIR")" && pwd)/$(basename "$DIR")"
fi

if [[ ! -d "$DIR/.git" ]]; then
  echo "Cloning noai-watermark into $DIR (pinned ref: $REF)"
  git clone --depth 1 --filter=blob:none --sparse \
    https://github.com/mertizci/noai-watermark.git "$DIR"
  git -C "$DIR" fetch --depth 1 origin "$REF"
  git -C "$DIR" checkout --detach "$REF"
  git -C "$DIR" sparse-checkout set --no-cone '/src/'
  HEAD_SHA="$(git -C "$DIR" rev-parse HEAD)"
  if [[ "$HEAD_SHA" != "$REF" ]]; then
    echo "error: expected pinned ref $REF, got $HEAD_SHA" >&2
    exit 1
  fi
else
  echo "Using existing checkout: $DIR"
  HEAD_SHA="$(git -C "$DIR" rev-parse HEAD 2>/dev/null || true)"
  if [[ "$HEAD_SHA" != "$REF" ]]; then
    echo "existing checkout not at pinned ref $REF (HEAD: ${HEAD_SHA:-missing}); re-pinning"
    git -C "$DIR" fetch --depth 1 origin "$REF" || {
      echo "error: could not fetch pinned ref $REF" >&2
      exit 1
    }
    git -C "$DIR" checkout --detach "$REF"
    git -C "$DIR" sparse-checkout set --no-cone '/src/'
    HEAD_SHA="$(git -C "$DIR" rev-parse HEAD)"
    if [[ "$HEAD_SHA" != "$REF" ]]; then
      echo "error: expected pinned ref $REF, got $HEAD_SHA" >&2
      exit 1
    fi
  fi
fi

if [[ ! -x "$DIR/.venv/bin/python" ]]; then
  echo "Creating venv at $DIR/.venv"
  "$PYTHON" -m venv "$DIR/.venv"
fi

echo "Installing Python dependencies"
# Pin pip itself (unpinned --upgrade pip was a supply-chain drift point).
"$DIR/.venv/bin/python" -m pip install --upgrade "pip==26.2.1"

# Install torch with the right platform index before the other pinned deps.
if command -v nvidia-smi >/dev/null 2>&1; then
  cuda="$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: \([0-9]*\.[0-9]*\).*/\1/p' | head -1)"
  if [[ -n "$cuda" ]]; then
    tag="cu${cuda/./}"
    index="https://download.pytorch.org/whl/$tag"
    echo "NVIDIA GPU detected (CUDA $cuda); installing torch from $index"
    "$DIR/.venv/bin/python" -m pip install torch --index-url "$index"
  else
    echo "nvidia-smi present but no CUDA version found; installing default torch"
    "$DIR/.venv/bin/python" -m pip install torch
  fi
else
  echo "No NVIDIA GPU detected; installing default torch (CPU/MPS)"
  "$DIR/.venv/bin/python" -m pip install torch
fi

"$DIR/.venv/bin/python" -m pip install -r "$SCRIPT_DIR/requirements-ctrlregen.txt"

cat <<EOF

Done. Remove a watermark with:

  export NOAI_WATERMARK_DIR="$DIR"
  "$DIR/.venv/bin/python" "$SCRIPT_DIR/clean_ctrlregen.py" IMAGE -o OUT
EOF
