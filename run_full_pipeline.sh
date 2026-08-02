#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/llama-3.2-3b-id-dr-stateft}"
DATA_PATH="${DATA_PATH:-datasets/commonsense_170k.json}"
TEST_DATA_PATH="${TEST_DATA_PATH:-datasets}"

"$PYTHON" train_id_dr_3b.py \
  --data-path "$DATA_PATH" \
  --output-dir "$OUTPUT_DIR" \
  "$@"

"$PYTHON" export_compact.py \
  --source "$OUTPUT_DIR" \
  --output "$OUTPUT_DIR/compact"

"$PYTHON" evaluate_3b.py \
  --output-dir "$OUTPUT_DIR" \
  --test-data-path "$TEST_DATA_PATH" \
  --results-file "$OUTPUT_DIR/evaluation.csv"

"$PYTHON" analyze_id_rank.py \
  --adapter-dir "$OUTPUT_DIR"
