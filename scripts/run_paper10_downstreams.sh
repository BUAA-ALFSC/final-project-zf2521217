#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

EVAL_STRIDE="${EVAL_STRIDE:-96}"
EVAL_SAMPLES="${EVAL_SAMPLES:-10}"
BATCH_SIZE="${BATCH_SIZE:-64}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-30}"
MAX_TRAIN_BATCHES="${MAX_TRAIN_BATCHES:-128}"
MAX_VAL_BATCHES="${MAX_VAL_BATCHES:-128}"
MAX_EVAL_WINDOWS="${MAX_EVAL_WINDOWS:-0}"
RUN_PAPER_BASELINES="${RUN_PAPER_BASELINES:-1}"
BASELINE_MAX_WINDOWS="${BASELINE_MAX_WINDOWS:-64}"
BASELINE_N_JOBS="${BASELINE_N_JOBS:-1}"
RUN_DEEP_BASELINES="${RUN_DEEP_BASELINES:-1}"
DEEP_BASELINE_EPOCHS="${DEEP_BASELINE_EPOCHS:-10}"
DEEP_BASELINE_BATCH_SIZE="${DEEP_BASELINE_BATCH_SIZE:-32}"
DEEP_BASELINE_MAX_TRAIN_BATCHES="${DEEP_BASELINE_MAX_TRAIN_BATCHES:-64}"
DEEP_BASELINE_MAX_VAL_BATCHES="${DEEP_BASELINE_MAX_VAL_BATCHES:-32}"
DEEP_BASELINE_MAX_EVAL_WINDOWS="${DEEP_BASELINE_MAX_EVAL_WINDOWS:-64}"
STUDENT_T_FIXED_DF="${STUDENT_T_FIXED_DF:-0}"

STUDENT_T_ARGS=()
if [[ "$STUDENT_T_FIXED_DF" == "1" ]]; then
  STUDENT_T_ARGS=(--student_t_fixed_df)
fi

PRETRAIN_CKPT="${PRETRAIN_CKPT:-results/checkpoints/lag_llama_paper10_pretrain_best.ckpt}"

run_eval() {
  local name="$1"
  local data_path="$2"
  local ckpt_path="$3"
  local metrics_file="$4"
  local figure_file="$5"

  python -m src.evaluate \
    --dataset csv \
    --data_path "$data_path" \
    --target_column OT \
    --time_column date \
    --checkpoint_path "$ckpt_path" \
    --output_checkpoint "$(basename "$ckpt_path")" \
    --metrics_file "$metrics_file" \
    --figure_file "$figure_file" \
    --device_target CPU \
    --num_samples "$EVAL_SAMPLES" \
    --batch_size 8 \
    --window_stride "$EVAL_STRIDE" \
    --max_eval_windows "$MAX_EVAL_WINDOWS" \
    --log_every 10 \
    "${STUDENT_T_ARGS[@]}"
}

run_paper_baselines() {
  local name="$1"
  local data_path="$2"

  python scripts/run_paper_baselines.py \
    --data_path "$data_path" \
    --target_column OT \
    --time_column date \
    --window_stride "$EVAL_STRIDE" \
    --max_windows "$BASELINE_MAX_WINDOWS" \
    --n_jobs "$BASELINE_N_JOBS" \
    --output_file "paper_baselines_${name}.json" \
    --figure_file "paper_baselines_${name}.png"
}

run_deep_baselines() {
  local name="$1"
  local data_path="$2"

  python scripts/run_deep_baselines.py \
    --dataset_name "$name" \
    --data_path "$data_path" \
    --target_column OT \
    --time_column date \
    --device_target Ascend \
    --epochs "$DEEP_BASELINE_EPOCHS" \
    --batch_size "$DEEP_BASELINE_BATCH_SIZE" \
    --window_stride "$EVAL_STRIDE" \
    --max_train_batches "$DEEP_BASELINE_MAX_TRAIN_BATCHES" \
    --max_val_batches "$DEEP_BASELINE_MAX_VAL_BATCHES" \
    --max_eval_windows "$DEEP_BASELINE_MAX_EVAL_WINDOWS" \
    --baselines DeepAR PatchTST \
    --output_file "paper_deep_baselines_${name}.json" \
    --figure_file "paper_deep_baselines_${name}.png"
}

run_finetune() {
  local name="$1"
  local data_path="$2"
  local output_ckpt="$3"

  python -m src.train \
    --dataset csv \
    --data_path "$data_path" \
    --target_column OT \
    --time_column date \
    --checkpoint_path "$PRETRAIN_CKPT" \
    --output_checkpoint "$output_ckpt" \
    --history_file "train_history_paper10_${name}_finetune.json" \
    --device_target Ascend \
    --epochs "$FINETUNE_EPOCHS" \
    --learning_rate 0.00001 \
    --batch_size "$BATCH_SIZE" \
    --max_train_batches "$MAX_TRAIN_BATCHES" \
    --max_val_batches "$MAX_VAL_BATCHES" \
    --early_stopping_patience 10 \
    --aug_prob 0.5 \
    --freq_mask_rate 0.5 \
    --freq_mixing_rate 0.25 \
    --log_every 10 \
    "${STUDENT_T_ARGS[@]}"
}

declare -A DATASETS=(
  [ettm2]="data/paper10_downstream/ETTm2.csv"
  [exchange_rate]="data/paper10_downstream/exchange_rate_000.csv"
  [weather]="data/paper10_downstream/weather_000.csv"
)

for name in ettm2 exchange_rate weather; do
  data_path="${DATASETS[$name]}"
  if [[ ! -f "$data_path" ]]; then
    echo "skip $name: missing $data_path"
    continue
  fi

  if [[ "$RUN_PAPER_BASELINES" == "1" ]]; then
    echo "paper baselines downstream=$name"
    run_paper_baselines "$name" "$data_path"
  fi

  if [[ "$RUN_DEEP_BASELINES" == "1" ]]; then
    echo "paper deep baselines downstream=$name"
    run_deep_baselines "$name" "$data_path"
  fi

  echo "zero-shot downstream=$name"
  run_eval \
    "$name" \
    "$data_path" \
    "$PRETRAIN_CKPT" \
    "metrics_paper10_zero_shot_${name}.json" \
    "forecast_paper10_zero_shot_${name}.png"

  output_ckpt="lag_llama_paper10_${name}_finetune.ckpt"
  echo "fine-tune downstream=$name"
  run_finetune "$name" "$data_path" "$output_ckpt"

  finetune_best="results/checkpoints/${output_ckpt%.ckpt}_best.ckpt"
  echo "fine-tuned evaluation downstream=$name"
  run_eval \
    "$name" \
    "$data_path" \
    "$finetune_best" \
    "metrics_paper10_finetuned_${name}.json" \
    "forecast_paper10_finetuned_${name}.png"
done

python scripts/plot_paper10_summary.py
