#!/usr/bin/env bash
# Full WAVES evaluation: iterate watermarks x engines x intensities.
# Aggregates results into results/eval_matrix.json.
#
# Usage:
#   ./benchmarks/run_full_eval.sh --waves-dir /path/to/WAVES --output-dir results/
#
# Requires: Python 3.12+, all engines installed (torch, diffusers, httpx, etc.)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
BENCH="$SCRIPT_DIR/bench_waves.py"

WAVES_DIR=""
OUTPUT_DIR="results"
DATASET="coco"
JSON_OUT=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --waves-dir) WAVES_DIR="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --dataset) DATASET="$2"; shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 2 ;;
    esac
done

if [[ -z "$WAVES_DIR" ]]; then
    echo "Usage: $0 --waves-dir <path> [--output-dir <path>] [--dataset <name>]" >&2
    exit 2
fi

WATERMARKS=(tree_ring gaussian_shading stegastamp stable_signature synthid hidden trustmark prc seal)
ENGINES=(marknull nfpa fmdiffwa synthid_bypass treering chain)
INTENSITIES=(0.3 0.5 0.7 1.0)

MATRIX_DIR="$OUTPUT_DIR/matrix"
mkdir -p "$MATRIX_DIR"

echo "WAVES full evaluation"
echo "  Watermarks: ${#WATERMARKS[@]}"
echo "  Engines:    ${#ENGINES[@]}"
echo "  Intensities: ${#INTENSITIES[@]}"
echo "  Total combos: $(( ${#WATERMARKS[@]} * ${#ENGINES[@]} * ${#INTENSITIES[@]} ))"
echo ""

RESULTS=()

for wm in "${WATERMARKS[@]}"; do
    for eng in "${ENGINES[@]}"; do
        for intensity in "${INTENSITIES[@]}"; do
            combo="${wm}_${eng}_${intensity}"
            out_dir="$MATRIX_DIR/$combo"
            echo -n "[$combo] ... "

            result=$(python "$BENCH" \
                --waves-dir "$WAVES_DIR" \
                --engine "$eng" \
                --watermark "$wm" \
                --intensity "$intensity" \
                --output-dir "$out_dir" \
                --dataset "$DATASET" \
                --json 2>/dev/null || echo '{"error": true}')

            echo "$result" > "$MATRIX_DIR/${combo}.json"
            RESULTS+=("$result")
            echo "done"
        done
    done
done

# Aggregate into eval_matrix.json
python -c "
import json, sys, pathlib

matrix_dir = pathlib.Path('$MATRIX_DIR')
results = []
for f in sorted(matrix_dir.glob('*.json')):
    try:
        data = json.loads(f.read_text())
        data['combo'] = f.stem
        results.append(data)
    except Exception:
        pass

out = pathlib.Path('$OUTPUT_DIR') / 'eval_matrix.json'
out.write_text(json.dumps(results, indent=2))
print(f'Wrote {len(results)} results to {out}')
"

echo ""
echo "Evaluation complete. Results: $OUTPUT_DIR/eval_matrix.json"
