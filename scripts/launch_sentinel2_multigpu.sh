#!/usr/bin/env bash
set -euo pipefail

# Launch Sentinel-2 → Clay embedding extraction across multiple GPUs.
#
# Default design:
# - 8 shards for 8 A5000 GPUs
# - NYC is split across 4 shards because it dominates the workload
# - Remaining cities are grouped by observed 2017–2025 row counts
#
# Usage:
#   bash scripts/launch_sentinel2_multigpu.sh
#   YEAR_MIN=2020 YEAR_MAX=2020 bash scripts/launch_sentinel2_multigpu.sh
#   GPU_IDS=0,1,2,3 bash scripts/launch_sentinel2_multigpu.sh   # edit shards first if using <8 GPUs
#
# Outputs:
#   data/processed/s2_shards/<shard>_metadata.parquet
#   data/processed/s2_shards/<shard>_embeddings.npy
#   logs/s2_shards/<shard>.log
#
# After all shards finish:
#   source <YOUR_CONDA_SETUP>
#   python scripts/merge_sentinel2_shards.py

ROOT="<REPO_ROOT>"
cd "$ROOT"

# Activate your Python environment before running this launcher.
# Example:
#   source ~/miniconda3/etc/profile.d/conda.sh
#   conda activate ghgbench

YEAR_MIN="${YEAR_MIN:-2017}"
YEAR_MAX="${YEAR_MAX:-2025}"
EE_PROJECT="${EE_PROJECT:-earth-engine-493114}"
CLAY_CKPT="${CLAY_CKPT:-data/models/v1.5/clay-v1.5.ckpt}"
CLAY_METADATA="${CLAY_METADATA:-data/models/clay_metadata.yaml}"

SHARD_DIR="data/processed/s2_shards"
LOG_DIR="logs/s2_shards"
mkdir -p "$SHARD_DIR" "$LOG_DIR"

# Comma-separated GPU IDs to use. Default assumes 8 visible GPUs.
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
IFS=',' read -r -a GPUS <<< "$GPU_IDS"

if [ "${#GPUS[@]}" -lt 8 ]; then
  echo "This launcher expects 8 GPU IDs. Got: ${#GPUS[@]} (${GPU_IDS})" >&2
  echo "Either provide 8 GPUs or edit the shard plan in this script." >&2
  exit 1
fi

declare -a SHARD_NAMES=(
  "nyc_2017_2018"
  "nyc_2019_2020"
  "nyc_2021_2022"
  "nyc_2023_2025"
  "la_only"
  "seattle_dc"
  "chicago_boston_sf"
  "rest_world"
)

declare -a SHARD_CITIES=(
  "nyc"
  "nyc"
  "nyc"
  "nyc"
  "la"
  "seattle,dc"
  "chicago,boston,sf"
  "sydney,denver,singapore,melbourne,brisbane,portland,perth,philadelphia,canberra,adelaide,gold_coast,hobart,darwin,newcastle,wollongong,townsville,cairns,geelong,port_macquarie"
)

declare -a SHARD_YEAR_MIN=(
  "2017"
  "2019"
  "2021"
  "2023"
  "${YEAR_MIN}"
  "${YEAR_MIN}"
  "${YEAR_MIN}"
  "${YEAR_MIN}"
)

declare -a SHARD_YEAR_MAX=(
  "2018"
  "2020"
  "2022"
  "${YEAR_MAX}"
  "${YEAR_MAX}"
  "${YEAR_MAX}"
  "${YEAR_MAX}"
  "${YEAR_MAX}"
)

echo "Launching ${#SHARD_NAMES[@]} shards with YEAR_MIN=${YEAR_MIN}, YEAR_MAX=${YEAR_MAX}"

for i in "${!SHARD_NAMES[@]}"; do
  shard="${SHARD_NAMES[$i]}"
  cities="${SHARD_CITIES[$i]}"
  shard_y0="${SHARD_YEAR_MIN[$i]}"
  shard_y1="${SHARD_YEAR_MAX[$i]}"
  y0=$(( shard_y0 > YEAR_MIN ? shard_y0 : YEAR_MIN ))
  y1=$(( shard_y1 < YEAR_MAX ? shard_y1 : YEAR_MAX ))
  if [ "$y0" -gt "$y1" ]; then
    echo
    echo "[skip] shard=${shard} cities=${cities} years=${shard_y0}-${shard_y1} outside requested ${YEAR_MIN}-${YEAR_MAX}"
    continue
  fi
  gpu="${GPUS[$i]}"
  meta_out="${SHARD_DIR}/${shard}_metadata.parquet"
  emb_out="${SHARD_DIR}/${shard}_embeddings.npy"
  log_out="${LOG_DIR}/${shard}.log"

  echo
  echo "[launch] shard=${shard} gpu=${gpu} cities=${cities} years=${y0}-${y1}"
  nohup env CUDA_VISIBLE_DEVICES="${gpu}" \
    python scripts/extract_sentinel2_embeddings.py \
      --cities "${cities}" \
      --year_min "${y0}" \
      --year_max "${y1}" \
      --device cuda \
      --clay_ckpt "${CLAY_CKPT}" \
      --clay_metadata "${CLAY_METADATA}" \
      --ee_project "${EE_PROJECT}" \
      --meta_out "${meta_out}" \
      --emb_out "${emb_out}" \
      > "${log_out}" 2>&1 &

  echo "  pid=$!  log=${log_out}"
  # Small stagger reduces the initial GEE request burst when many shards start together.
  sleep $((i * 5))
done

echo
echo "All shard jobs launched."
echo "Monitor with:"
echo "  ls ${LOG_DIR}"
echo "  tail -f ${LOG_DIR}/nyc_2019_2020.log"
