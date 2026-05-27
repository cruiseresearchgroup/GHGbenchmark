#!/bin/bash
# Run Task A experiments in separate batches (one per feature set).
#
# Two-layer benchmark (2026-04-12 redesign):
#   Layer 1 — core_all_cities  (26 cities, location + climate + size)
#            core_all_cities_climate_plus (same 26, + 4 extra climate channels)
#   Layer 2A — us_core, us_metadata, us_leaky_eui, us_leaky_full (6 US cities)
#   Layer 2B — au_core, au_eui, au_full (15 Australian cities)
#
# Each feature set writes to its own CSV (task_a_results_<fs>.csv) so running
# batches separately does NOT overwrite previous results. After all batches
# finish, run `bash scripts/run_task_a_all.sh merge` to concatenate them.
#
# Usage:
#   bash scripts/run_task_a_all.sh                     # run all registered feature sets sequentially then merge
#   bash scripts/run_task_a_all.sh core_all_cities     # run one feature set
#   bash scripts/run_task_a_all.sh us_core             # ...
#   bash scripts/run_task_a_all.sh au_core             # ...
#   bash scripts/run_task_a_all.sh merge               # merge per-fs CSVs into final CSV
#
# Each feature set batch: 6 classical models × 2 split types × 2 modes = 24 experiments.
# GPU MLP is run as a separate backfill track and merged later when desired.
# Results are written incrementally — a crash mid-run preserves finished experiments.

set -e

cd <REPO_ROOT>

PYTHON="python"
MODELS="GlobalMean,CityTypeMean,Ridge,RandomForest,XGBoost,LightGBM"
SPLIT_TYPES="random,grouped"
SEEDS="${TASK_A_SEEDS:-42,123,456}"
SCRIPT="scripts/run_task_a.py"
# All outputs go to clean_building/ (the root results/ task_a_results*.csv paths
# are a historical merge-polluted bundle — see building_final_summary_zh.md §0).
RESULTS_DIR="results/clean_building"
mkdir -p "${RESULTS_DIR}"

ALL_FS="core_all_cities core_all_cities_climate_plus us_core us_metadata us_leaky_eui us_leaky_full au_core au_eui au_full"

run_one() {
    local FS=$1
    local OUT="${RESULTS_DIR}/task_a_results_${FS}.csv"
    echo ""
    echo "=========================================="
    echo "=== Task A: feature_set=${FS}"
    echo "=== Output: ${OUT}"
    echo "=== Seeds: ${SEEDS}"
    echo "=========================================="
    PYTHONUNBUFFERED=1 $PYTHON -u $SCRIPT \
        --feature_sets "$FS" \
        --models "$MODELS" \
        --seeds "$SEEDS" \
        --split_types "$SPLIT_TYPES" \
        --mode both \
        --out_path "$OUT"
    echo "=== Done: ${FS} ==="
}

merge_results() {
    echo "Merging per-feature-set CSVs into ${RESULTS_DIR}/task_a_results.csv..."
    $PYTHON -c "
import os
import pandas as pd
feature_sets = '${ALL_FS}'.split()
files = [os.path.join('${RESULTS_DIR}', f'task_a_results_{fs}.csv') for fs in feature_sets]
files = [f for f in files if os.path.exists(f)]
if not files:
    print('No per-fs files found.')
    exit(1)
dfs = [pd.read_csv(f) for f in files]
out = pd.concat(dfs, ignore_index=True)
out_path = os.path.join('${RESULTS_DIR}', 'task_a_results.csv')
out.to_csv(out_path, index=False)
print(f'Merged {len(files)} files -> {out_path} ({len(out)} rows)')
for f in files:
    print(f'  - {f}: {len(pd.read_csv(f))} rows')
"
}

case "$1" in
    core_all_cities|core_all_cities_climate_plus|us_core|us_metadata|us_leaky_eui|us_leaky_full|au_core|au_eui|au_full)
        run_one "$1"
        ;;
    merge)
        merge_results
        ;;
    "")
        for FS in $ALL_FS; do
            run_one "$FS"
        done
        merge_results
        ;;
    *)
        echo "Usage: $0 [core_all_cities|core_all_cities_climate_plus|us_core|us_metadata|us_leaky_eui|us_leaky_full|au_core|au_eui|au_full|merge]"
        exit 1
        ;;
esac
