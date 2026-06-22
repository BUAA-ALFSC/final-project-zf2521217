from __future__ import annotations

from dataclasses import dataclass
import gzip
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass
class WindowArrays:
    past_target: np.ndarray
    past_observed_values: np.ndarray
    future_target: np.ndarray
    future_observed_values: np.ndarray
    past_time_feat: np.ndarray | None = None
    future_time_feat: np.ndarray | None = None


@dataclass
class SplitData:
    train: WindowArrays
    val: WindowArrays
    test: WindowArrays
    raw_series: np.ndarray


def concat_window_arrays(parts: list[WindowArrays]) -> WindowArrays:
    if not parts:
        raise ValueError("No window arrays to concatenate.")
    return WindowArrays(
        past_target=np.concatenate([p.past_target for p in parts], axis=0),
        past_observed_values=np.concatenate([p.past_observed_values for p in parts], axis=0),
        future_target=np.concatenate([p.future_target for p in parts], axis=0),
        future_observed_values=np.concatenate([p.future_observed_values for p in parts], axis=0),
        past_time_feat=np.concatenate([p.past_time_feat for p in parts], axis=0)
        if parts[0].past_time_feat is not None
        else None,
        future_time_feat=np.concatenate([p.future_time_feat for p in parts], axis=0)
        if parts[0].future_time_feat is not None
        else None,
    )


def generate_synthetic_series(num_points: int, seed: int = 42) -> np.ndarray:
    """Smoke-test data only. Final experiments should use paper datasets."""
    rng = np.random.default_rng(seed)
    t = np.arange(num_points, dtype=np.float32)
    series = (
        np.sin(2 * np.pi * t / 48.0)
        + 0.45 * np.sin(2 * np.pi * t / 168.0)
        + 0.0015 * t
        + 0.08 * rng.standard_normal(num_points)
    )
    return series.astype(np.float32)


def make_time_features(
    timestamps: pd.Series | pd.DatetimeIndex | None,
    length: int,
    freq: str = "15min",
) -> np.ndarray:
    if timestamps is None:
        start = pd.Timestamp("2000-01-01")
        timestamps = pd.date_range(start=start, periods=length, freq=freq)
    else:
        timestamps = pd.to_datetime(timestamps)

    idx = pd.DatetimeIndex(timestamps)
    features = np.stack(
        [
            idx.minute.to_numpy(dtype=np.float32) / 59.0 - 0.5,
            idx.hour.to_numpy(dtype=np.float32) / 23.0 - 0.5,
            idx.dayofweek.to_numpy(dtype=np.float32) / 6.0 - 0.5,
            (idx.day.to_numpy(dtype=np.float32) - 1.0) / 30.0 - 0.5,
            (idx.dayofyear.to_numpy(dtype=np.float32) - 1.0) / 365.0 - 0.5,
            (idx.month.to_numpy(dtype=np.float32) - 1.0) / 11.0 - 0.5,
        ],
        axis=-1,
    )
    return features.astype(np.float32)


def load_csv_series(
    path: str | Path,
    target_column: str = "",
    time_column: str = "date",
    freq: str = "15min",
) -> tuple[np.ndarray, np.ndarray]:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV dataset not found: {csv_path}")

    frame = pd.read_csv(csv_path)
    if target_column:
        if target_column not in frame.columns:
            raise ValueError(
                f"target_column={target_column!r} not found. Columns: {list(frame.columns)}"
            )
        values = frame[target_column].to_numpy(dtype=np.float32)
    else:
        numeric = frame.select_dtypes(include=[np.number])
        if numeric.empty:
            raise ValueError(f"No numeric columns found in {csv_path}")
        values = numeric.iloc[:, -1].to_numpy(dtype=np.float32)

    if time_column and time_column in frame.columns:
        time_features = make_time_features(frame[time_column], len(frame), freq=freq)
    else:
        time_features = make_time_features(None, len(frame), freq=freq)
    finite_mask = np.isfinite(values)
    values = values[finite_mask]
    time_features = time_features[finite_mask]
    if values.size == 0:
        raise ValueError(f"No finite target values found in {csv_path}")
    return values.astype(np.float32), time_features


def _read_json_records(path: Path) -> list[dict]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        if "data" in payload:
            return payload["data"]
        if "train" in payload and "test" in payload:
            return payload["train"] + payload["test"]
        if "target" in payload:
            return [payload]
    raise ValueError(f"Unsupported JSON dataset structure: {path}")


def load_json_series(
    path: str | Path,
    series_index: int = 0,
    freq: str = "15min",
) -> tuple[np.ndarray, np.ndarray]:
    dataset_path = Path(path)
    if dataset_path.is_dir():
        candidates = [
            dataset_path / "test" / "data.json.gz",
            dataset_path / "train" / "data.json.gz",
            dataset_path / "data.json.gz",
            dataset_path / "data.json",
        ]
        for candidate in candidates:
            if candidate.exists():
                dataset_path = candidate
                break
        else:
            raise FileNotFoundError(
                f"No supported JSON/GZip dataset file found under {dataset_path}"
            )

    records = _read_json_records(dataset_path)
    numeric_records = []
    for record in records:
        target = np.asarray(record.get("target", []), dtype=np.float32)
        if target.size and np.all(np.isfinite(target)):
            numeric_records.append(target)
    if not numeric_records:
        raise ValueError(f"No numeric target records found in {dataset_path}")
    if series_index < 0 or series_index >= len(numeric_records):
        raise IndexError(
            f"series_index={series_index} out of range for {len(numeric_records)} series"
        )
    series = numeric_records[series_index].astype(np.float32)
    return series, make_time_features(None, len(series), freq=freq)


def make_windows(
    series: np.ndarray,
    time_features: np.ndarray | None,
    context_length: int,
    prediction_length: int,
    max_lag: int,
    start: int,
    end: int,
    stride: int = 1,
) -> WindowArrays:
    past_length = context_length + max_lag
    total_length = past_length + prediction_length
    if end - start < total_length:
        raise ValueError(
            f"Not enough points for windows: available={end - start}, need={total_length}"
        )

    past, future = [], []
    past_tf, future_tf = [], []
    for idx in range(start, end - total_length + 1, stride):
        window = series[idx : idx + total_length]
        past.append(window[:past_length])
        future.append(window[past_length:])
        if time_features is not None:
            tf_window = time_features[idx : idx + total_length]
            past_tf.append(tf_window[:past_length])
            future_tf.append(tf_window[past_length:])

    past_target = np.asarray(past, dtype=np.float32)
    future_target = np.asarray(future, dtype=np.float32)
    past_time_feat = np.asarray(past_tf, dtype=np.float32) if past_tf else None
    future_time_feat = np.asarray(future_tf, dtype=np.float32) if future_tf else None
    return WindowArrays(
        past_target=past_target,
        past_observed_values=np.ones_like(past_target, dtype=np.float32),
        future_target=future_target,
        future_observed_values=np.ones_like(future_target, dtype=np.float32),
        past_time_feat=past_time_feat,
        future_time_feat=future_time_feat,
    )


def split_series_windows(
    series: np.ndarray,
    time_features: np.ndarray | None,
    context_length: int,
    prediction_length: int,
    max_lag: int,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    stride: int = 1,
) -> SplitData:
    n = len(series)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    past_length = context_length + max_lag

    # Overlap split starts by past_length so validation/test have enough history.
    train = make_windows(series, time_features, context_length, prediction_length, max_lag, 0, train_end, stride=stride)
    val = make_windows(
        series,
        time_features,
        context_length,
        prediction_length,
        max_lag,
        max(0, train_end - past_length),
        val_end,
        stride=stride,
    )
    test = make_windows(
        series,
        time_features,
        context_length,
        prediction_length,
        max_lag,
        max(0, val_end - past_length),
        n,
        stride=stride,
    )
    return SplitData(train=train, val=val, test=test, raw_series=series)


def iter_batches(data: WindowArrays, batch_size: int, shuffle: bool, seed: int = 42) -> Iterable[dict]:
    indices = np.arange(data.past_target.shape[0])
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)
    for start in range(0, len(indices), batch_size):
        idx = indices[start : start + batch_size]
        yield {
            "past_target": data.past_target[idx],
            "past_observed_values": data.past_observed_values[idx],
            "future_target": data.future_target[idx],
            "future_observed_values": data.future_observed_values[idx],
            "past_time_feat": data.past_time_feat[idx] if data.past_time_feat is not None else None,
            "future_time_feat": data.future_time_feat[idx] if data.future_time_feat is not None else None,
        }


def load_experiment_data(
    dataset: str,
    data_path: str,
    train_data_paths: tuple[str, ...] | list[str],
    val_data_path: str,
    test_data_path: str,
    target_column: str,
    time_column: str,
    freq: str,
    context_length: int,
    prediction_length: int,
    max_lag: int,
    train_ratio: float,
    val_ratio: float,
    seed: int,
    num_synthetic_points: int,
    series_index: int = 0,
    window_stride: int = 1,
) -> SplitData:
    if dataset == "synthetic":
        series = generate_synthetic_series(num_synthetic_points, seed=seed)
        time_features = make_time_features(None, len(series), freq=freq)
    elif dataset == "csv":
        series, time_features = load_csv_series(
            data_path, target_column=target_column, time_column=time_column, freq=freq
        )
    elif dataset == "json":
        series, time_features = load_json_series(data_path, series_index=series_index, freq=freq)
    elif dataset == "multi_csv":
        train_parts = []
        raw_series_parts = []
        for path in train_data_paths:
            train_series, train_time_features = load_csv_series(
                path, target_column=target_column, time_column=time_column, freq=freq
            )
            raw_series_parts.append(train_series)
            train_parts.append(
                make_windows(
                    train_series,
                    train_time_features,
                    context_length=context_length,
                    prediction_length=prediction_length,
                    max_lag=max_lag,
                    start=0,
                    end=len(train_series),
                    stride=window_stride,
                )
            )
        if val_data_path:
            val_series, val_time_features = load_csv_series(
                val_data_path, target_column=target_column, time_column=time_column, freq=freq
            )
        else:
            val_series = raw_series_parts[-1]
            val_time_features = make_time_features(None, len(val_series), freq=freq)
        if test_data_path:
            test_series, test_time_features = load_csv_series(
                test_data_path, target_column=target_column, time_column=time_column, freq=freq
            )
        else:
            test_series = val_series
            test_time_features = val_time_features
        val_test = split_series_windows(
            val_series,
            val_time_features,
            context_length=context_length,
            prediction_length=prediction_length,
            max_lag=max_lag,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            stride=window_stride,
        )
        if test_data_path and test_data_path != val_data_path:
            test_split = split_series_windows(
                test_series,
                test_time_features,
                context_length=context_length,
                prediction_length=prediction_length,
                max_lag=max_lag,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                stride=window_stride,
            )
            test = test_split.test
        else:
            test = val_test.test
        return SplitData(
            train=concat_window_arrays(train_parts),
            val=val_test.val,
            test=test,
            raw_series=np.concatenate(raw_series_parts),
        )
    elif dataset in {"csv_dir", "multi_csv_split"}:
        if train_data_paths:
            csv_paths = [Path(path) for path in train_data_paths]
        else:
            data_dir = Path(data_path)
            if not data_dir.exists():
                raise FileNotFoundError(f"CSV directory not found: {data_dir}")
            csv_paths = sorted(data_dir.glob("*.csv"))
        if not csv_paths:
            raise FileNotFoundError(
                "No CSV files found for csv_dir/multi_csv_split dataset. "
                "Set data_path to a directory or pass train_data_paths."
            )

        train_parts = []
        val_parts = []
        test_parts = []
        raw_series_parts = []
        for path in csv_paths:
            series_part, time_features_part = load_csv_series(
                path, target_column=target_column, time_column=time_column, freq=freq
            )
            try:
                split = split_series_windows(
                    series_part,
                    time_features_part,
                    context_length=context_length,
                    prediction_length=prediction_length,
                    max_lag=max_lag,
                    train_ratio=train_ratio,
                    val_ratio=val_ratio,
                    stride=window_stride,
                )
            except ValueError as exc:
                print(f"skip {path}: {exc}")
                continue
            train_parts.append(split.train)
            val_parts.append(split.val)
            test_parts.append(split.test)
            raw_series_parts.append(series_part)
        return SplitData(
            train=concat_window_arrays(train_parts),
            val=concat_window_arrays(val_parts),
            test=concat_window_arrays(test_parts),
            raw_series=np.concatenate(raw_series_parts),
        )
    else:
        raise ValueError(
            "Supported dataset values are 'synthetic', 'csv', 'multi_csv', "
            "'csv_dir', 'multi_csv_split', and 'json'. Use csv/multi_csv/csv_dir "
            "for ETT-style files, or json for GluonTS/ListDataset-style files."
        )
    return split_series_windows(
        series,
        time_features,
        context_length=context_length,
        prediction_length=prediction_length,
        max_lag=max_lag,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        stride=window_stride,
    )
