from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    import mindspore as ms
    import mindspore.nn as nn
    from mindspore import Tensor
except ModuleNotFoundError as exc:  # pragma: no cover - environment guidance
    raise SystemExit(
        "MindSpore is not installed in this Python environment. "
        "Run this script on ModelArts or a MindSpore environment."
    ) from exc


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.baseline_models import DeepARBaseline, ForecastMSELoss, PatchTSTBaseline
from src.config import CHECKPOINT_DIR, FIGURES_DIR, RESULTS_DIR, ExperimentConfig
from src.data import iter_batches, load_experiment_data
from src.device import configure_device
from src.metrics import mae, mse
from src.train import set_seed, time_feature_tensor


PAPER_DEEP_BASELINES = ("DeepAR", "PatchTST")


def select_windows(data, max_windows: int):
    if max_windows <= 0 or max_windows >= data.past_target.shape[0]:
        return data
    indices = np.linspace(0, data.past_target.shape[0] - 1, max_windows, dtype=np.int64)
    indices = np.unique(indices)
    data.past_target = data.past_target[indices]
    data.past_observed_values = data.past_observed_values[indices]
    data.future_target = data.future_target[indices]
    data.future_observed_values = data.future_observed_values[indices]
    if data.past_time_feat is not None:
        data.past_time_feat = data.past_time_feat[indices]
    if data.future_time_feat is not None:
        data.future_time_feat = data.future_time_feat[indices]
    return data


def build_model(name: str, context_length: int, prediction_length: int, use_time_features: bool):
    input_size = 1 + (6 if use_time_features else 0)
    if name == "DeepAR":
        return DeepARBaseline(
            input_size=input_size,
            hidden_size=64,
            prediction_length=prediction_length,
            num_layers=2,
        )
    if name == "PatchTST":
        return PatchTSTBaseline(
            context_length=context_length,
            prediction_length=prediction_length,
            patch_length=8,
            stride=4,
            d_model=64,
            nhead=4,
            num_layers=2,
        )
    raise ValueError(f"Unsupported deep baseline: {name}")


def predict_batches(model, data, batch_size: int, seed: int) -> np.ndarray:
    model.set_train(False)
    pred_parts = []
    for batch in iter_batches(data, batch_size, shuffle=False, seed=seed):
        past = Tensor(batch["past_target"], ms.float32)
        past_tf = time_feature_tensor(batch["past_time_feat"], past.shape[0], past.shape[1])
        pred_parts.append(model(past, past_tf).asnumpy())
    return np.concatenate(pred_parts, axis=0)


def evaluate_model(model, data, batch_size: int, seed: int) -> dict:
    pred = predict_batches(model, data, batch_size, seed)
    return {
        "mae": mae(data.future_target, pred),
        "mse": mse(data.future_target, pred),
    }


def train_one_baseline(name: str, split, cfg: ExperimentConfig, args: argparse.Namespace) -> dict:
    model = build_model(name, cfg.context_length + cfg.max_lag, cfg.prediction_length, cfg.use_time_features)
    loss_net = ForecastMSELoss(model)
    optimizer = nn.Adam(model.trainable_params(), learning_rate=args.learning_rate, weight_decay=args.weight_decay)
    grad_fn = ms.value_and_grad(loss_net, None, optimizer.parameters)

    def train_step(batch: dict):
        past = Tensor(batch["past_target"], ms.float32)
        future = Tensor(batch["future_target"], ms.float32)
        past_tf = time_feature_tensor(batch["past_time_feat"], past.shape[0], past.shape[1])
        loss, grads = grad_fn(past, past_tf, future)
        optimizer(grads)
        return loss

    history = {
        "train_loss": [],
        "val_loss": [],
        "best_val_loss": [],
        "best_epoch": 0,
        "early_stopped": False,
    }
    best_val_loss = float("inf")
    best_epoch = 0
    no_improve = 0
    start_time = time.time()
    stem = f"paper_baseline_{args.dataset_name}_{name.lower()}".replace("-", "_")
    best_ckpt = CHECKPOINT_DIR / f"{stem}_best.ckpt"

    for epoch in range(1, args.epochs + 1):
        model.set_train(True)
        train_losses = []
        for batch_idx, batch in enumerate(iter_batches(split.train, cfg.batch_size, shuffle=True, seed=cfg.seed + epoch), start=1):
            loss = float(train_step(batch).asnumpy())
            train_losses.append(loss)
            if args.log_every and batch_idx % args.log_every == 0:
                print(f"{name} epoch={epoch:03d} batch={batch_idx:05d} train_loss={loss:.6f}", flush=True)
            if cfg.max_train_batches and batch_idx >= cfg.max_train_batches:
                break

        val_scores = evaluate_model(model, split.val, cfg.batch_size, cfg.seed)
        train_loss = float(np.mean(train_losses))
        val_loss = float(val_scores["mse"])
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            no_improve = 0
            ms.save_checkpoint(model, str(best_ckpt))
        else:
            no_improve += 1
        history["best_val_loss"].append(best_val_loss)
        history["best_epoch"] = best_epoch
        print(
            f"{name} epoch={epoch:03d} train_loss={train_loss:.6f} "
            f"val_mse={val_loss:.6f} best_val_mse={best_val_loss:.6f} best_epoch={best_epoch:03d}",
            flush=True,
        )
        if args.early_stopping_patience and no_improve >= args.early_stopping_patience:
            history["early_stopped"] = True
            history["stopped_epoch"] = epoch
            break

    if best_ckpt.exists():
        ms.load_checkpoint(str(best_ckpt), net=model)
    test_scores = evaluate_model(model, split.test, cfg.batch_size, cfg.seed)
    test_scores.update(
        {
            "status": "ok",
            "implementation": f"MindSpore supervised {name} baseline",
            "checkpoint": str(best_ckpt),
            "history": history,
            "elapsed_sec": time.time() - start_time,
        }
    )
    return test_scores


def plot_deep_baselines(metrics: dict, output_path: Path) -> None:
    names = []
    values = []
    for name in PAPER_DEEP_BASELINES:
        entry = metrics.get(name, {})
        if entry.get("status") == "ok":
            names.append(name)
            values.append(float(entry["mae"]))
    if not names:
        print("skip deep baseline plot: no successful deep baselines")
        return
    x = np.arange(len(names))
    plt.figure(figsize=(6.5, 4.0))
    bars = plt.bar(x, values)
    plt.xticks(x, names)
    plt.ylabel("MAE")
    plt.title("Paper Deep Baseline MAE")
    plt.grid(axis="y", alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, f"{height:.3f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close()
    print(f"saved {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--dataset", default="csv")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--target_column", default="OT")
    parser.add_argument("--time_column", default="date")
    parser.add_argument("--freq", default="15min")
    parser.add_argument("--context_length", type=int, default=32)
    parser.add_argument("--prediction_length", type=int, default=24)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--window_stride", type=int, default=96)
    parser.add_argument("--max_train_batches", type=int, default=64)
    parser.add_argument("--max_val_batches", type=int, default=64)
    parser.add_argument("--max_eval_windows", type=int, default=64)
    parser.add_argument("--early_stopping_patience", type=int, default=3)
    parser.add_argument("--device_target", default="Ascend")
    parser.add_argument("--baselines", nargs="+", default=list(PAPER_DEEP_BASELINES))
    parser.add_argument("--output_file", default="")
    parser.add_argument("--figure_file", default="")
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--no_time_features", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("MPLCONFIGDIR", str(RESULTS_DIR / ".matplotlib"))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    device = configure_device(args.device_target)
    set_seed(42)

    cfg = ExperimentConfig(
        dataset=args.dataset,
        data_path=args.data_path,
        target_column=args.target_column,
        time_column=args.time_column,
        freq=args.freq,
        context_length=args.context_length,
        prediction_length=args.prediction_length,
        batch_size=args.batch_size,
        window_stride=args.window_stride,
        max_train_batches=args.max_train_batches,
        max_val_batches=args.max_val_batches,
        use_time_features=not args.no_time_features,
        device_target=device,
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
        max_lag=cfg.max_lag,
        train_ratio=cfg.train_ratio,
        val_ratio=cfg.val_ratio,
        seed=cfg.seed,
        num_synthetic_points=cfg.num_synthetic_points,
        series_index=cfg.series_index,
        window_stride=cfg.window_stride,
    )
    split.train = select_windows(split.train, args.max_eval_windows)
    split.val = select_windows(split.val, min(args.max_eval_windows, args.max_val_batches or args.max_eval_windows))
    split.test = select_windows(split.test, args.max_eval_windows)
    print(
        "deep_baseline_start "
        f"dataset={args.dataset_name} train_windows={split.train.past_target.shape[0]} "
        f"val_windows={split.val.past_target.shape[0]} test_windows={split.test.past_target.shape[0]} "
        f"baselines={args.baselines}",
        flush=True,
    )

    metrics: dict[str, dict] = {}
    for name in args.baselines:
        if name not in PAPER_DEEP_BASELINES:
            metrics[name] = {"status": "skipped", "reason": f"Unsupported deep baseline {name}"}
            continue
        try:
            print(f"deep_baseline_run name={name}", flush=True)
            metrics[name] = train_one_baseline(name, split, cfg, args)
            print(f"deep_baseline_done name={name} status={metrics[name].get('status')}", flush=True)
        except Exception as exc:  # pragma: no cover - backend dependent
            metrics[name] = {
                "status": "failed",
                "implementation": f"MindSpore supervised {name} baseline",
                "reason": f"{type(exc).__name__}: {exc}",
            }
            print(f"deep_baseline_done name={name} status=failed reason={type(exc).__name__}: {exc}", flush=True)

    available = {name: entry for name, entry in metrics.items() if entry.get("status") == "ok"}
    if available:
        best_name = min(available, key=lambda name: available[name]["mae"])
        best = available[best_name]
    else:
        best_name = ""
        best = {}

    payload = {
        "config": {
            "dataset_name": args.dataset_name,
            "data_path": args.data_path,
            "context_length": args.context_length,
            "prediction_length": args.prediction_length,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "window_stride": args.window_stride,
            "max_train_batches": args.max_train_batches,
            "max_val_batches": args.max_val_batches,
            "max_eval_windows": args.max_eval_windows,
            "device_target": device,
        },
        "paper_deep_baseline_metrics": metrics,
        "best_available_deep_baseline": best_name,
        "best_available_deep_baseline_mae": best.get("mae"),
        "best_available_deep_baseline_mse": best.get("mse"),
    }
    output_file = args.output_file or f"paper_deep_baselines_{args.dataset_name}.json"
    output_path = RESULTS_DIR / output_file
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"deep_baseline_metrics={output_path}", flush=True)

    figure_file = args.figure_file or f"paper_deep_baselines_{args.dataset_name}.png"
    plot_deep_baselines(metrics, FIGURES_DIR / figure_file)


if __name__ == "__main__":
    main()
