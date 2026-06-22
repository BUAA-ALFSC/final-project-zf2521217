from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import FIGURES_DIR, RESULTS_DIR, ExperimentConfig
from src.data import load_experiment_data
from src.metrics import mae, mse


PAPER_STAT_BASELINES = ("AutoETS", "DynOptTheta")
UNSELECTED_PAPER_BASELINES = (
    "AutoARIMA",
    "CrostonSBA",
    "NPTS",
    "TFT",
    "N-BEATS",
    "Informer",
    "AutoFormer",
    "ETSFormer",
    "OneFitsAll",
)


def infer_season_length(data_path: str, default: int) -> int:
    name = Path(data_path).name.lower()
    if "ettm" in name:
        return 96
    if "etth" in name or "weather" in name:
        return 24
    if "exchange" in name:
        return 7
    return default


def infer_freq(data_path: str, requested: str) -> str:
    if requested:
        return requested
    name = Path(data_path).name.lower()
    if "ettm" in name:
        return "15min"
    if "etth" in name or "weather" in name:
        return "H"
    if "exchange" in name:
        return "D"
    return "D"


def select_windows(
    past: np.ndarray,
    future: np.ndarray,
    max_windows: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if max_windows <= 0 or max_windows >= past.shape[0]:
        indices = np.arange(past.shape[0])
        return past, future, indices
    indices = np.linspace(0, past.shape[0] - 1, max_windows, dtype=np.int64)
    indices = np.unique(indices)
    return past[indices], future[indices], indices


def build_statsforecast_frame(past: np.ndarray, freq: str) -> tuple[pd.DataFrame, list[str]]:
    frames = []
    item_ids = []
    for idx, series in enumerate(past):
        item_id = f"window_{idx:06d}"
        item_ids.append(item_id)
        timestamps = pd.date_range("2000-01-01", periods=series.shape[0], freq=freq)
        frames.append(
            pd.DataFrame(
                {
                    "unique_id": item_id,
                    "ds": timestamps,
                    "y": series.astype(float),
                }
            )
        )
    return pd.concat(frames, ignore_index=True), item_ids


def statsforecast_model(name: str, season_length: int):
    from statsforecast.models import AutoETS

    try:
        from statsforecast.models import DynamicOptimizedTheta
    except ImportError:  # pragma: no cover - depends on statsforecast version
        from statsforecast.models import DynamicTheta as DynamicOptimizedTheta

    if name == "AutoETS":
        return AutoETS(season_length=season_length)
    if name == "DynOptTheta":
        return DynamicOptimizedTheta(season_length=season_length)
    raise ValueError(f"Unsupported statsforecast baseline: {name}")


def run_statsforecast_baseline(
    name: str,
    past: np.ndarray,
    future: np.ndarray,
    prediction_length: int,
    freq: str,
    season_length: int,
    n_jobs: int,
) -> tuple[dict, np.ndarray | None]:
    try:
        from statsforecast import StatsForecast
    except ModuleNotFoundError as exc:
        return {
            "status": "skipped",
            "reason": "statsforecast is not installed",
            "install": "pip install statsforecast",
        }, None

    start_time = time.time()
    try:
        frame, item_ids = build_statsforecast_frame(past, freq)
        model = statsforecast_model(name, season_length)
        forecaster = StatsForecast(models=[model], freq=freq, n_jobs=n_jobs)
        forecast_frame = forecaster.forecast(df=frame, h=prediction_length)
        value_columns = [col for col in forecast_frame.columns if col not in {"unique_id", "ds"}]
        if not value_columns:
            raise RuntimeError(f"No forecast column returned by statsforecast for {name}")
        column = value_columns[0]
        predictions = []
        for item_id in item_ids:
            values = forecast_frame.loc[forecast_frame["unique_id"] == item_id, column].to_numpy(dtype=np.float32)
            if values.shape[0] != prediction_length:
                raise RuntimeError(
                    f"{name} returned horizon={values.shape[0]} for {item_id}, expected={prediction_length}"
                )
            predictions.append(values)
        pred = np.stack(predictions, axis=0)
        if not np.all(np.isfinite(pred)):
            raise RuntimeError(f"{name} returned non-finite forecasts")
        return {
            "status": "ok",
            "implementation": "statsforecast",
            "mae": mae(future, pred),
            "mse": mse(future, pred),
            "elapsed_sec": time.time() - start_time,
        }, pred
    except Exception as exc:  # pragma: no cover - depends on optional library internals
        return {
            "status": "failed",
            "implementation": "statsforecast",
            "reason": f"{type(exc).__name__}: {exc}",
            "elapsed_sec": time.time() - start_time,
        }, None


def plot_baseline_bars(metrics: dict, output_path: Path) -> None:
    names = []
    values = []
    for name in PAPER_STAT_BASELINES:
        entry = metrics.get(name, {})
        if entry.get("status") == "ok":
            names.append(name)
            values.append(float(entry["mae"]))
    if not names:
        print("skip baseline plot: no paper baseline metrics available")
        return

    x = np.arange(len(names))
    plt.figure(figsize=(8.5, 4.2))
    bars = plt.bar(x, values)
    plt.xticks(x, names, rotation=20, ha="right")
    plt.ylabel("MAE")
    plt.title("Selected Paper Statistical Baseline MAE")
    plt.grid(axis="y", alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close()
    print(f"saved {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="csv")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--target_column", default="OT")
    parser.add_argument("--time_column", default="date")
    parser.add_argument("--freq", default="")
    parser.add_argument("--context_length", type=int, default=32)
    parser.add_argument("--prediction_length", type=int, default=24)
    parser.add_argument("--max_lag", type=int, default=673)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--window_stride", type=int, default=96)
    parser.add_argument("--max_windows", type=int, default=0)
    parser.add_argument("--season_length", type=int, default=0)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--n_jobs", type=int, default=1)
    parser.add_argument("--output_file", default="paper_baselines.json")
    parser.add_argument("--figure_file", default="paper_baselines.png")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    freq = infer_freq(args.data_path, args.freq)
    season_length = args.season_length or infer_season_length(args.data_path, default=24)
    cfg = ExperimentConfig(
        dataset=args.dataset,
        data_path=args.data_path,
        target_column=args.target_column,
        time_column=args.time_column,
        freq=freq,
        context_length=args.context_length,
        prediction_length=args.prediction_length,
        window_stride=args.window_stride,
    )

    split = load_experiment_data(
        dataset=cfg.dataset,
        data_path=cfg.data_path,
        train_data_paths=cfg.train_data_paths,
        val_data_path=cfg.val_data_path,
        test_data_path=cfg.test_data_path,
        target_column=cfg.target_column,
        time_column=cfg.time_column,
        freq=cfg.freq,
        context_length=cfg.context_length,
        prediction_length=cfg.prediction_length,
        max_lag=args.max_lag,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=cfg.seed,
        num_synthetic_points=cfg.num_synthetic_points,
        series_index=cfg.series_index,
        window_stride=cfg.window_stride,
    )
    past, future, indices = select_windows(split.test.past_target, split.test.future_target, args.max_windows)
    print(
        "paper_baseline_start "
        f"dataset={Path(args.data_path).name} windows={past.shape[0]} "
        f"prediction_length={args.prediction_length} freq={freq} season_length={season_length}",
        flush=True,
    )

    metrics: dict[str, dict] = {}
    predictions: dict[str, np.ndarray] = {}
    for name in PAPER_STAT_BASELINES:
        print(f"paper_baseline_run name={name}", flush=True)
        result, pred = run_statsforecast_baseline(
            name,
            past,
            future,
            args.prediction_length,
            freq,
            season_length,
            args.n_jobs,
        )
        metrics[name] = result
        if pred is not None:
            predictions[name] = pred
        print(f"paper_baseline_done name={name} status={result.get('status')}", flush=True)

    for name in UNSELECTED_PAPER_BASELINES:
        metrics[name] = {
            "status": "not_run",
            "reason": "Not selected for the 2-statistical + 2-deep baseline budget plan.",
        }

    available = {name: entry for name, entry in metrics.items() if entry.get("status") == "ok"}
    if available:
        best_name = min(available, key=lambda name: available[name]["mae"])
        best = available[best_name]
    else:
        best_name = ""
        best = {}

    payload = {
        "config": {
            "data_path": args.data_path,
            "target_column": args.target_column,
            "time_column": args.time_column,
            "freq": freq,
            "context_length": args.context_length,
            "prediction_length": args.prediction_length,
            "max_lag": args.max_lag,
            "window_stride": args.window_stride,
            "evaluated_windows": int(past.shape[0]),
            "selected_window_indices": indices.tolist(),
            "season_length": season_length,
        },
        "paper_baseline_metrics": metrics,
        "selection_note": "Runs selected paper statistical baselines: AutoETS and DynOptTheta.",
        "best_available_paper_baseline": best_name,
        "best_available_paper_baseline_mae": best.get("mae"),
        "best_available_paper_baseline_mse": best.get("mse"),
    }

    output_path = RESULTS_DIR / args.output_file
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"paper_baseline_metrics={output_path}", flush=True)

    plot_baseline_bars(metrics, FIGURES_DIR / args.figure_file)


if __name__ == "__main__":
    main()
