from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import load_csv_series, load_json_series


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["csv", "json"], required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--target_column", default="")
    parser.add_argument("--series_index", type=int, default=0)
    args = parser.parse_args()

    if not Path(args.data_path).exists():
        raise SystemExit(f"Dataset path not found: {args.data_path}")

    if args.dataset == "csv":
        frame = pd.read_csv(args.data_path, nrows=5)
        print("columns:", list(frame.columns))
        series, time_features = load_csv_series(args.data_path, args.target_column)
    else:
        series, time_features = load_json_series(args.data_path, args.series_index)

    print("length:", len(series))
    print("time_features_shape:", tuple(time_features.shape))
    print("min:", float(np.min(series)))
    print("max:", float(np.max(series)))
    print("mean:", float(np.mean(series)))
    print("first_10:", series[:10].tolist())


if __name__ == "__main__":
    main()
