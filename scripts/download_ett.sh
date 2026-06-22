#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p data

BASE_URL="https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small"
FILES=(ETTh1.csv ETTh2.csv ETTm1.csv ETTm2.csv)

for file in "${FILES[@]}"; do
  url="${BASE_URL}/${file}"
  out="data/${file}"
  if [ -f "$out" ]; then
    echo "$out already exists"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "$out" "$url"
  else
    curl -L "$url" -o "$out"
  fi
  python scripts/inspect_dataset.py --dataset csv --data_path "$out"
done

