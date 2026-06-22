#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-100}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-64}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-10}"
AUG_PROB="${AUG_PROB:-0.5}"
FREQ_MASK_RATE="${FREQ_MASK_RATE:-0.5}"
FREQ_MIXING_RATE="${FREQ_MIXING_RATE:-0.25}"
STUDENT_T_FIXED_DF="${STUDENT_T_FIXED_DF:-0}"
STUDENT_T_ARGS=()
if [[ "$STUDENT_T_FIXED_DF" == "1" ]]; then
  STUDENT_T_ARGS=(--student_t_fixed_df)
fi

bash scripts/download_ett.sh

python -m src.train \
  --ett_pretrain \
  --epochs "$PRETRAIN_EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --early_stopping_patience "$EARLY_STOPPING_PATIENCE" \
  --aug_prob "$AUG_PROB" \
  --freq_mask_rate "$FREQ_MASK_RATE" \
  --freq_mixing_rate "$FREQ_MIXING_RATE" \
  "${STUDENT_T_ARGS[@]}"

python -m src.evaluate \
  --ett_zero_shot \
  --batch_size "$BATCH_SIZE" \
  --device_target CPU \
  "${STUDENT_T_ARGS[@]}"

python -m src.train \
  --ett_finetune \
  --epochs "$FINETUNE_EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --early_stopping_patience "$EARLY_STOPPING_PATIENCE" \
  --aug_prob "$AUG_PROB" \
  --freq_mask_rate "$FREQ_MASK_RATE" \
  --freq_mixing_rate "$FREQ_MIXING_RATE" \
  "${STUDENT_T_ARGS[@]}"

python -m src.evaluate \
  --ett_finetune \
  --batch_size "$BATCH_SIZE" \
  --device_target CPU \
  "${STUDENT_T_ARGS[@]}"
