from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
PRETRAIN_DIR = DATA_DIR / "paper10_pretrain"
DOWNSTREAM_DIR = DATA_DIR / "paper10_downstream"

ETT_URLS = {
    "ETTh1": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv",
    "ETTh2": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh2.csv",
    "ETTm1": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm1.csv",
    "ETTm2": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm2.csv",
}

# These names are from the Lag-Llama paper/pretraining script. They are loaded
# through GluonTS's public dataset repository, not synthetically generated.
DEFAULT_GLUONTS_PRETRAIN = (
    "electricity_hourly",
    "solar_10_minutes",
    "traffic",
    "kdd_cup_2018_without_missing",
    "sunspot_without_missing",
    "australian_electricity_demand",
    "london_smart_meters_without_missing",
)
DEFAULT_GLUONTS_DOWNSTREAM = (
    "exchange_rate",
    "weather",
)


def download(url: str, output_path: Path) -> None:
    if output_path.exists() and output_path.stat().st_size > 0:
        print(f"{output_path} already exists")
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"download {url} -> {output_path}")
    urllib.request.urlretrieve(url, output_path)


def convert_ett(source: Path, target: Path) -> None:
    frame = pd.read_csv(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame[["date", "OT"]].to_csv(target, index=False)
    print(f"saved {target} rows={len(frame)}")


def infer_pandas_freq(freq: str) -> str:
    normalized = str(freq).upper()
    mapping = {
        "T": "min",
        "MIN": "min",
        "H": "h",
        "D": "D",
        "B": "B",
        "W": "W",
        "M": "M",
        "Q": "Q",
        "0.5H": "30min",
        "10T": "10min",
        "15T": "15min",
    }
    return mapping.get(normalized, freq)


def export_gluonts_dataset(
    name: str,
    output_dir: Path,
    max_series: int,
    max_points: int,
) -> int:
    try:
        from gluonts.dataset.repository.datasets import get_dataset
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "GluonTS is required to export real paper datasets. Install it on "
            "ModelArts with: pip install gluonts==0.14.4"
        ) from exc

    print(f"loading GluonTS dataset: {name}")
    dataset = get_dataset(name, path=DATA_DIR / "gluonts_cache")
    freq = infer_pandas_freq(dataset.metadata.freq)
    exported = 0
    for series_idx, item in enumerate(dataset.train):
        if exported >= max_series:
            break
        target = np.asarray(item["target"], dtype=np.float32)
        target = target[np.isfinite(target)]
        if max_points > 0 and len(target) > max_points:
            target = target[-max_points:]
        if len(target) < 720:
            continue
        start = pd.Period(item["start"]).to_timestamp() if hasattr(item["start"], "to_timestamp") else pd.Timestamp(item["start"])
        try:
            dates = pd.date_range(start=start, periods=len(target), freq=freq)
        except Exception:
            dates = pd.date_range(start="2000-01-01", periods=len(target), freq=freq)
        frame = pd.DataFrame({"date": dates, "OT": target})
        out = output_dir / f"{name}_{exported:03d}.csv"
        frame.to_csv(out, index=False)
        exported += 1
    print(f"exported {exported} series from {name}")
    return exported


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gluonts_datasets",
        nargs="*",
        default=list(DEFAULT_GLUONTS_PRETRAIN),
        help="Paper dataset names to export through GluonTS.",
    )
    parser.add_argument(
        "--downstream_gluonts_datasets",
        nargs="*",
        default=list(DEFAULT_GLUONTS_DOWNSTREAM),
        help="Paper downstream dataset names to export through GluonTS.",
    )
    parser.add_argument("--max_series_per_dataset", type=int, default=8)
    parser.add_argument("--max_points_per_series", type=int, default=20000)
    parser.add_argument(
        "--ettm2_in_pretrain",
        action="store_true",
        help="Include ETTm2 in pretraining. Disabled by default because ETTm2 is the downstream test dataset.",
    )
    args = parser.parse_args()

    PRETRAIN_DIR.mkdir(parents=True, exist_ok=True)
    DOWNSTREAM_DIR.mkdir(parents=True, exist_ok=True)

    raw_dir = DATA_DIR / "raw_ett"
    for name, url in ETT_URLS.items():
        download(url, raw_dir / f"{name}.csv")

    for name in ("ETTh1", "ETTh2", "ETTm1"):
        convert_ett(raw_dir / f"{name}.csv", PRETRAIN_DIR / f"{name}.csv")
    if args.ettm2_in_pretrain:
        convert_ett(raw_dir / "ETTm2.csv", PRETRAIN_DIR / "ETTm2.csv")
    convert_ett(raw_dir / "ETTm2.csv", DOWNSTREAM_DIR / "ETTm2.csv")

    total_gluonts = 0
    for dataset_name in args.gluonts_datasets:
        total_gluonts += export_gluonts_dataset(
            dataset_name,
            PRETRAIN_DIR,
            max_series=args.max_series_per_dataset,
            max_points=args.max_points_per_series,
        )
    total_downstream = 0
    for dataset_name in args.downstream_gluonts_datasets:
        total_downstream += export_gluonts_dataset(
            dataset_name,
            DOWNSTREAM_DIR,
            max_series=args.max_series_per_dataset,
            max_points=args.max_points_per_series,
        )

    csv_count = len(list(PRETRAIN_DIR.glob("*.csv")))
    if csv_count < 10:
        print(
            f"warning: only {csv_count} pretraining CSV files were prepared. "
            "Check network access or reduce dataset selection.",
            file=sys.stderr,
        )
    print(f"prepared real paper-dataset subset CSV count={csv_count}")
    print(f"prepared downstream GluonTS CSV count={total_downstream}")
    print(f"pretrain_dir={PRETRAIN_DIR}")
    print(f"downstream_dir={DOWNSTREAM_DIR}")


if __name__ == "__main__":
    main()
