#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
STUDENT_T_FIXED_DF="${STUDENT_T_FIXED_DF:-0}"
STUDENT_T_ARGS=()
if [[ "$STUDENT_T_FIXED_DF" == "1" ]]; then
  STUDENT_T_ARGS=(--student_t_fixed_df)
fi

python -m src.train --smoke "${STUDENT_T_ARGS[@]}"
python -m src.evaluate --smoke --num_samples 10 "${STUDENT_T_ARGS[@]}"
