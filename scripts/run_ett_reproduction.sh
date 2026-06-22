#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

STUDENT_T_FIXED_DF="${STUDENT_T_FIXED_DF:-0}"
STUDENT_T_ARGS=()
if [[ "$STUDENT_T_FIXED_DF" == "1" ]]; then
  STUDENT_T_ARGS=(--student_t_fixed_df)
fi

bash scripts/download_ett.sh

python -m src.train --ett_pretrain "${STUDENT_T_ARGS[@]}"

python -m src.evaluate --ett_zero_shot --device_target CPU "${STUDENT_T_ARGS[@]}"

python -m src.train --ett_finetune "${STUDENT_T_ARGS[@]}"

python -m src.evaluate --ett_finetune --device_target CPU "${STUDENT_T_ARGS[@]}"
