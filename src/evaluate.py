from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    import mindspore as ms
    from mindspore import Tensor
except ModuleNotFoundError as exc:  # pragma: no cover - environment guidance
    raise SystemExit(
        "MindSpore is not installed in this Python environment. "
        "Use ModelArts MindSpore image or create a supported local Python environment."
    ) from exc

from .config import (
    CHECKPOINT_DIR,
    FIGURES_DIR,
    RESULTS_DIR,
    ExperimentConfig,
    ett_finetune_config,
    ett_quick_config,
    ett_zero_shot_config,
    paper10_finetune_config,
    paper10_zero_shot_config,
    smoke_config,
)
from .data import iter_batches, load_experiment_data
from .device import configure_device
from .metrics import classical_baseline_forecasts, mae, mean_w_quantile_loss, mse, sample_crps
from .model import LagLlamaMindSpore
from .train import build_model_config, time_feature_tensor


def evaluate(cfg: ExperimentConfig) -> dict:
    os.environ.setdefault("MPLCONFIGDIR", str(RESULTS_DIR / ".matplotlib"))
    cfg.device_target = configure_device(cfg.device_target)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

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
        max_lag=cfg.max_lag,
        train_ratio=cfg.train_ratio,
        val_ratio=cfg.val_ratio,
        seed=cfg.seed,
        num_synthetic_points=cfg.num_synthetic_points,
        series_index=cfg.series_index,
        window_stride=cfg.window_stride,
    )

    model = LagLlamaMindSpore(build_model_config(cfg))
    ckpt_path = Path(cfg.checkpoint_path) if cfg.checkpoint_path else CHECKPOINT_DIR / cfg.output_checkpoint
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ms.load_checkpoint(str(ckpt_path), net=model)
    model.set_train(False)

    past = split.test.past_target
    future = split.test.future_target
    if cfg.max_eval_windows and cfg.max_eval_windows < future.shape[0]:
        indices = np.linspace(0, future.shape[0] - 1, cfg.max_eval_windows, dtype=np.int64)
        indices = np.unique(indices)
        past = past[indices]
        future = future[indices]
        split.test.past_target = past
        split.test.past_observed_values = split.test.past_observed_values[indices]
        split.test.future_target = future
        split.test.future_observed_values = split.test.future_observed_values[indices]
        if split.test.past_time_feat is not None:
            split.test.past_time_feat = split.test.past_time_feat[indices]
        if split.test.future_time_feat is not None:
            split.test.future_time_feat = split.test.future_time_feat[indices]
    pred_parts = []
    sample_parts = []
    total_windows = int(future.shape[0])
    total_batches = int(np.ceil(total_windows / cfg.batch_size))
    print(
        "eval_start "
        f"windows={total_windows} batch_size={cfg.batch_size} "
        f"batches={total_batches} prediction_length={cfg.prediction_length} "
        f"num_samples={cfg.num_samples}",
        flush=True,
    )
    start_time = time.time()
    for batch_idx, batch in enumerate(iter_batches(split.test, cfg.batch_size, shuffle=False, seed=cfg.seed), start=1):
        batch_past = batch["past_target"]
        batch_future = batch["future_target"]
        batch_observed = batch["past_observed_values"]
        past_time_feat = time_feature_tensor(batch["past_time_feat"], batch_past.shape[0], batch_past.shape[1])
        future_time_feat = time_feature_tensor(batch["future_time_feat"], batch_future.shape[0], batch_future.shape[1])
        pred_parts.append(
            model.predict_mean(
                Tensor(batch_past, ms.float32),
                Tensor(batch_observed, ms.float32),
                cfg.prediction_length,
                past_time_feat=past_time_feat,
                future_time_feat=future_time_feat,
            ).asnumpy()
        )
        sample_parts.append(
            model.predict_samples(
                Tensor(batch_past, ms.float32),
                Tensor(batch_observed, ms.float32),
                cfg.prediction_length,
                cfg.num_samples,
                past_time_feat=past_time_feat,
                future_time_feat=future_time_feat,
            ).asnumpy()
        )
        if cfg.log_every and (batch_idx == 1 or batch_idx % cfg.log_every == 0 or batch_idx == total_batches):
            elapsed = time.time() - start_time
            batches_per_sec = batch_idx / elapsed if elapsed > 0 else 0.0
            print(
                "eval_progress "
                f"batch={batch_idx:05d}/{total_batches:05d} "
                f"elapsed_sec={elapsed:.1f} batches_per_sec={batches_per_sec:.3f}",
                flush=True,
            )

    pred = np.concatenate(pred_parts, axis=0)
    samples = np.concatenate(sample_parts, axis=0)
    baseline_forecasts = classical_baseline_forecasts(
        past,
        cfg.prediction_length,
        context_length=cfg.context_length,
        seasonal_periods=(24, 96, 168, 672),
    )
    last_value_baseline = baseline_forecasts["last_value"]
    baseline_scores = {
        name: {
            "mae": mae(future, forecast),
            "mse": mse(future, forecast),
        }
        for name, forecast in baseline_forecasts.items()
    }
    best_baseline_name = min(baseline_scores, key=lambda name: baseline_scores[name]["mae"])
    best_baseline = baseline_forecasts[best_baseline_name]

    model_mean_wql = mean_w_quantile_loss(future, samples)
    metrics = {
        "model_mae": mae(future, pred),
        "model_mse": mse(future, pred),
        "model_crps": model_mean_wql,
        "model_mean_wQuantileLoss": model_mean_wql,
        "model_sample_crps": sample_crps(future, samples),
        "naive_mae": baseline_scores["last_value"]["mae"],
        "naive_mse": baseline_scores["last_value"]["mse"],
        "best_classical_baseline": best_baseline_name,
        "best_classical_baseline_mae": baseline_scores[best_baseline_name]["mae"],
        "best_classical_baseline_mse": baseline_scores[best_baseline_name]["mse"],
    }
    for name, scores in baseline_scores.items():
        metrics[f"baseline_{name}_mae"] = scores["mae"]
        metrics[f"baseline_{name}_mse"] = scores["mse"]
    with open(RESULTS_DIR / cfg.metrics_file, "w", encoding="utf-8") as f:
        json.dump({"config": cfg.to_dict(), "metrics": metrics}, f, indent=2)
    with open(RESULTS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"config": cfg.to_dict(), "metrics": metrics}, f, indent=2)

    idx = 0
    history = past[idx, -cfg.context_length :]
    truth = future[idx]
    forecast = pred[idx]
    naive = last_value_baseline[idx]
    best_classical = best_baseline[idx]
    x_hist = np.arange(len(history))
    x_future = np.arange(len(history), len(history) + cfg.prediction_length)

    plt.figure(figsize=(10, 4))
    plt.plot(x_hist, history, label="history")
    plt.plot(x_future, truth, label="ground truth")
    plt.plot(x_future, forecast, label="Lag-Llama MindSpore")
    plt.plot(x_future, naive, label="last-value baseline", linestyle="--")
    if best_baseline_name != "last_value":
        plt.plot(x_future, best_classical, label=f"best classical: {best_baseline_name}", linestyle=":")
    plt.legend()
    plt.tight_layout()
    fig_path = FIGURES_DIR / cfg.figure_file
    plt.savefig(fig_path, dpi=160)
    plt.close()
    metrics["figure"] = str(fig_path)
    print(f"eval_done metrics={RESULTS_DIR / cfg.metrics_file} figure={fig_path}", flush=True)
    return metrics


def parse_args() -> ExperimentConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--ett_quick", action="store_true")
    parser.add_argument("--ett_zero_shot", action="store_true")
    parser.add_argument("--ett_finetune", action="store_true")
    parser.add_argument("--paper10_zero_shot", action="store_true")
    parser.add_argument("--paper10_finetune", action="store_true")
    parser.add_argument("--dataset", default="")
    parser.add_argument("--data_path", default="")
    parser.add_argument("--train_data_paths", nargs="*", default=None)
    parser.add_argument("--val_data_path", default="")
    parser.add_argument("--test_data_path", default="")
    parser.add_argument("--target_column", default="")
    parser.add_argument("--time_column", default="")
    parser.add_argument("--freq", default="")
    parser.add_argument("--series_index", type=int, default=None)
    parser.add_argument("--checkpoint_path", default="")
    parser.add_argument("--output_checkpoint", default="")
    parser.add_argument("--metrics_file", default="")
    parser.add_argument("--figure_file", default="")
    parser.add_argument("--loss", choices=["student_t", "mse"], default=None)
    parser.add_argument("--student_t_fixed_df", action="store_true")
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--device_target", default="")
    parser.add_argument("--context_length", type=int, default=None)
    parser.add_argument("--prediction_length", type=int, default=None)
    parser.add_argument("--window_stride", type=int, default=None)
    parser.add_argument("--max_eval_windows", type=int, default=None)
    parser.add_argument("--log_every", type=int, default=None)
    parser.add_argument("--no_time_features", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        cfg = smoke_config()
    elif args.ett_quick:
        cfg = ett_quick_config()
    elif args.ett_zero_shot:
        cfg = ett_zero_shot_config()
    elif args.ett_finetune:
        cfg = ett_finetune_config()
    elif args.paper10_zero_shot:
        cfg = paper10_zero_shot_config()
    elif args.paper10_finetune:
        cfg = paper10_finetune_config()
    else:
        cfg = ExperimentConfig()
    if args.dataset:
        cfg.dataset = args.dataset
    if args.data_path:
        cfg.data_path = args.data_path
    if args.train_data_paths is not None:
        cfg.train_data_paths = tuple(args.train_data_paths)
    if args.val_data_path:
        cfg.val_data_path = args.val_data_path
    if args.test_data_path:
        cfg.test_data_path = args.test_data_path
    if args.target_column:
        cfg.target_column = args.target_column
    if args.time_column:
        cfg.time_column = args.time_column
    if args.freq:
        cfg.freq = args.freq
    if args.series_index is not None:
        cfg.series_index = args.series_index
    if args.checkpoint_path:
        cfg.checkpoint_path = args.checkpoint_path
    if args.output_checkpoint:
        cfg.output_checkpoint = args.output_checkpoint
    if args.metrics_file:
        cfg.metrics_file = args.metrics_file
    if args.figure_file:
        cfg.figure_file = args.figure_file
    if args.loss is not None:
        cfg.loss = args.loss
    if args.student_t_fixed_df:
        cfg.student_t_fixed_df = True
    if args.num_samples is not None:
        cfg.num_samples = args.num_samples
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.device_target:
        cfg.device_target = args.device_target
    if args.context_length is not None:
        cfg.context_length = args.context_length
    if args.prediction_length is not None:
        cfg.prediction_length = args.prediction_length
    if args.window_stride is not None:
        cfg.window_stride = args.window_stride
    if args.max_eval_windows is not None:
        cfg.max_eval_windows = args.max_eval_windows
    if args.log_every is not None:
        cfg.log_every = args.log_every
    if args.no_time_features:
        cfg.use_time_features = False
    return cfg


def main() -> None:
    metrics = evaluate(parse_args())
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
