#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON="${PYTHON:-python3}"

TMP_DIR="$(mktemp -d 2>/dev/null || mktemp -d -t 'wm-smoke')"
trap 'rm -rf "$TMP_DIR"' EXIT

echo "Running smoke checks against test fixtures..."
"$PYTHON" "$SCRIPT_DIR/clean_text.py" "$ROOT_DIR/tests/fixtures/sample_watermarked.txt" -o "$TMP_DIR/wm.cleaned.txt" --stats
"$PYTHON" "$SCRIPT_DIR/rewrite_text.py" "$ROOT_DIR/tests/fixtures/sample_watermarked.txt" --backend print-prompt > /dev/null
"$PYTHON" "$SCRIPT_DIR/clean_file.py" "$ROOT_DIR/tests/fixtures/sample_ai.md" -o "$TMP_DIR/sample_ai.cleaned.md"
"$PYTHON" "$SCRIPT_DIR/clean_file.py" "$ROOT_DIR/tests/fixtures/sample_ai.html" -o "$TMP_DIR/sample_ai.cleaned.html"
"$PYTHON" "$SCRIPT_DIR/clean_file.py" "$ROOT_DIR/tests/fixtures/sample_meta.svg" -o "$TMP_DIR/sample_meta.cleaned.svg"
echo "Smoke checks passed."
