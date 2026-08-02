#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-python}"

for METHOD in fixed loss_exchange id_exchange; do
  OUTPUT_DIR="checkpoints/ablation-${METHOD}"
  "$PYTHON" train_id_dr_3b.py \
    --allocation-method "$METHOD" \
    --output-dir "$OUTPUT_DIR" \
    "$@"
  "$PYTHON" export_compact.py --source "$OUTPUT_DIR"
  "$PYTHON" evaluate_3b.py \
    --output-dir "$OUTPUT_DIR" \
    --results-file "$OUTPUT_DIR/evaluation.csv"
done
