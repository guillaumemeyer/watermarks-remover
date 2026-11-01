#!/usr/bin/env bash
set -euo pipefail

# Bootstrap the optional MarkDiffusion image-watermark harness backend.
#
# THU-BPM/MarkDiffusion (https://github.com/THU-BPM/MarkDiffusion) is
# Apache-2.0 and is NOT bundled in this repository. By default this script
# creates a venv and installs the package (plus the model deps it needs) from
# PyPI at a pinned version. Contributors can use --checkout to install an
# editable checkout of the upstream repo at a pinned commit instead.
#
# torch is installed separately so the right platform wheel index (CUDA or CPU)
# is used; markdiffusion's own torch range is then already satisfied.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_DIR="${MARKDIFFUSION_DIR:-$HOME/markdiffusion}"
DIR=""
# Pinned upstream commit for --checkout mode (v1.0.2, peeled tag commit). Do not
# point at a moving branch.
REF="cefdb320890acec728e3495b48824607d30329d3"
# Pinned PyPI release (also pinned in requirements-markdiffusion.txt).
PYPI_VERSION="1.0.2"
PYTHON="${PYTHON:-python3}"
CHECKOUT=0

usage() {
  cat <<EOF
Usage: setup_markdiffusion.sh [--dir PATH] [--ref REF] [--checkout] [--python PYTHON]

Creates a venv at <dir>/.venv and installs MarkDiffusion for the optional
markdiffusion_harness.py backend.

Options:
  --dir PATH     venv (and optional checkout) directory (default: \$MARKDIFFUSION_DIR or ~/markdiffusion)
  --checkout     install an editable checkout of THU-BPM/MarkDiffusion at a pinned commit
  --ref REF      git ref for --checkout (default: pinned commit SHA)
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
    --checkout)
      CHECKOUT=1
      shift
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

if [[ "$CHECKOUT" -eq 1 ]]; then
  if [[ ! -d "$DIR/.git" ]]; then
    echo "Cloning THU-BPM/MarkDiffusion into $DIR (pinned ref: $REF)"
    git clone --depth 1 --filter=blob:none --sparse \
      https://github.com/THU-BPM/MarkDiffusion.git "$DIR"
    git -C "$DIR" fetch --depth 1 origin "$REF"
    git -C "$DIR" checkout --detach "$REF"
    HEAD_SHA="$(git -C "$DIR" rev-parse HEAD)"
    if [[ "$HEAD_SHA" != "$REF" ]]; then
      echo "error: expected pinned ref $REF, got $HEAD_SHA" >&2
      exit 1
    fi
  else
    echo "Using existing checkout: $DIR"
  fi
fi

if [[ ! -x "$DIR/.venv/bin/python" ]]; then
  echo "Creating venv at $DIR/.venv"
  "$PYTHON" -m venv "$DIR/.venv"
fi

echo "Installing Python dependencies"
# Pin pip itself (unpinned --upgrade pip was a supply-chain drift point).
"$DIR/.venv/bin/python" -m pip install --upgrade "pip==26.2.1"

# Install torch with the right platform index before the rest. markdiffusion
# pins torch>=2.4,<2.11, so this satisfies its range and pip won't re-resolve.
if command -v nvidia-smi >/dev/null 2>&1; then
  cuda="$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: \([0-9]*\.[0-9]*\).*/\1/p' | head -1)"
  if [[ -n "$cuda" ]]; then
    tag="cu${cuda/./}"
    index="https://download.pytorch.org/whl/$tag"
    echo "NVIDIA GPU detected (CUDA $cuda); installing torch from $index"
    "$DIR/.venv/bin/python" -m pip install "torch>=2.4,<2.11" --index-url "$index"
  else
    echo "nvidia-smi present but no CUDA version found; installing default torch"
    "$DIR/.venv/bin/python" -m pip install "torch>=2.4,<2.11"
  fi
else
  echo "No NVIDIA GPU detected; installing default torch (CPU/MPS)"
  "$DIR/.venv/bin/python" -m pip install "torch>=2.4,<2.11"
fi

if [[ "$CHECKOUT" -eq 1 ]]; then
  echo "Installing MarkDiffusion editable checkout"
  "$DIR/.venv/bin/python" -m pip install -e "$DIR"
else
  echo "Installing MarkDiffusion $PYPI_VERSION from PyPI"
  "$DIR/.venv/bin/python" -m pip install -r "$SCRIPT_DIR/requirements-markdiffusion.txt"
fi

cat <<EOF

Done. Use the harness with:

  export MARKDIFFUSION_DIR="$DIR"
  "$DIR/.venv/bin/python" "$SCRIPT_DIR/markdiffusion_harness.py" --help
EOF
