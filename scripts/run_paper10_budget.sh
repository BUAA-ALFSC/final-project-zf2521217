#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-150}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-64}"
MAX_TRAIN_BATCHES="${MAX_TRAIN_BATCHES:-256}"
MAX_VAL_BATCHES="${MAX_VAL_BATCHES:-256}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-15}"
EVAL_STRIDE="${EVAL_STRIDE:-96}"
EVAL_SAMPLES="${EVAL_SAMPLES:-10}"
MAX_SERIES_PER_DATASET="${MAX_SERIES_PER_DATASET:-8}"
MAX_POINTS_PER_SERIES="${MAX_POINTS_PER_SERIES:-20000}"
STUDENT_T_FIXED_DF="${STUDENT_T_FIXED_DF:-0}"

STUDENT_T_ARGS=()
if [[ "$STUDENT_T_FIXED_DF" == "1" ]]; then
  STUDENT_T_ARGS=(--student_t_fixed_df)
fi

python scripts/prepare_paper10_datasets.py \
  --max_series_per_dataset "$MAX_SERIES_PER_DATASET" \
  --max_points_per_series "$MAX_POINTS_PER_SERIES"

python -m src.train \
  --paper10_pretrain \
  --device_target Ascend \
  --epochs "$PRETRAIN_EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --max_train_batches "$MAX_TRAIN_BATCHES" \
  --max_val_batches "$MAX_VAL_BATCHES" \
  --early_stopping_patience "$EARLY_STOPPING_PATIENCE" \
  --log_every 10 \
  "${STUDENT_T_ARGS[@]}"

EVAL_STRIDE="$EVAL_STRIDE" \
EVAL_SAMPLES="$EVAL_SAMPLES" \
BATCH_SIZE="$BATCH_SIZE" \
FINETUNE_EPOCHS="$FINETUNE_EPOCHS" \
MAX_TRAIN_BATCHES="$MAX_TRAIN_BATCHES" \
MAX_VAL_BATCHES="$MAX_VAL_BATCHES" \
STUDENT_T_FIXED_DF="$STUDENT_T_FIXED_DF" \
bash scripts/run_paper10_downstreams.sh
