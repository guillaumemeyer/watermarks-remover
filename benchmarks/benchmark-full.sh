#!/usr/bin/env bash
# Full SynthID-text benchmark: 10 docs x 3 seeds, three rewrite variants,
# re-stamp control, plus the minimal-rewrite-level scan. Variants land in
# out/bench-full/, minimal in out/bench-minimal/ (override with OUT_DIR).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
set -a; [ -f "$ROOT/.env" ] && . "$ROOT/.env"; set +a
export MARKLLM_DIR="${MARKLLM_DIR:-$HOME/MarkLLM}"
export HF_HOME="${HF_HOME:-$ROOT/.hf-cache}"

# Named-strength variants: does each rewrite remove the mark?
python3 "$ROOT/service/scripts/bench_synthid_text.py" \
  --markllm-dir "$MARKLLM_DIR" \
  --docs 10 --seeds 3 \
  --variants "paraphrase:1,paraphrase:3,backtranslate:1" \
  --restamp-control \
  --out-dir "${OUT_DIR:-$ROOT/out/bench-full}" \
  --tag full

# Minimal-rewrite-level scan: what is the smallest rewrite that removes the mark?
python3 "$ROOT/service/scripts/bench_synthid_text.py" \
  --markllm-dir "$MARKLLM_DIR" \
  --docs 10 --seeds 3 \
  --mode minimal \
  --rewrite-level-start 0.1 --rewrite-level-step 0.1 --rewrite-level-max 1.0 \
  --level-attempts 3 \
  --out-dir "${OUT_DIR:-$ROOT/out/bench-minimal}" \
  --tag minimal
