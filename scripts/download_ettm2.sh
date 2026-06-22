#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p data

URL="https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm2.csv"
OUT="data/ETTm2.csv"

if command -v wget >/dev/null 2>&1; then
  wget -O "$OUT" "$URL"
else
  curl -L "$URL" -o "$OUT"
fi

python scripts/inspect_dataset.py --dataset csv --data_path "$OUT"
