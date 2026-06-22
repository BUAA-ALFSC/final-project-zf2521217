from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    import mindspore as ms
    from mindspore import Tensor
except ModuleNotFoundError as exc:
    raise SystemExit(
        "MindSpore is not installed in this Python environment. "
        "Run this script on ModelArts or a MindSpore environment."
    ) from exc

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import FIGURES_DIR, ett_finetune_config, ett_zero_shot_config
from src.data import iter_batches, load_experiment_data
from src.device import configure_device
from src.metrics import naive_last_value_forecast
from src.model import LagLlamaMindSpore
from src.train import build_model_config, time_feature_tensor


def predict_all(cfg, checkpoint_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    ckpt = Path(checkpoint_path)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    ms.load_checkpoint(str(ckpt), net=model)
    model.set_train(False)

    pred_parts = []
    total_batches = int(np.ceil(split.test.future_target.shape[0] / cfg.batch_size))
    for batch_idx, batch in enumerate(iter_batches(split.test, cfg.batch_size, shuffle=False, seed=cfg.seed), start=1):
        batch_past = batch["past_target"]
        batch_future = batch["future_target"]
        batch_observed = batch["past_observed_values"]
        past_time_feat = time_feature_tensor(batch["past_time_feat"], batch_past.shape[0], batch_past.shape[1])
        future_time_feat = time_feature_tensor(batch["future_time_feat"], batch_future.shape[0], batch_future.shape[1])
        pred = model.predict_mean(
            Tensor(batch_past, ms.float32),
            Tensor(batch_observed, ms.float32),
            cfg.prediction_length,
            past_time_feat=past_time_feat,
            future_time_feat=future_time_feat,
        ).asnumpy()
        pred_parts.append(pred)
        if batch_idx == 1 or batch_idx % cfg.log_every == 0 or batch_idx == total_batches:
            print(f"{Path(checkpoint_path).name}: predicted batch {batch_idx}/{total_batches}", flush=True)

    pred = np.concatenate(pred_parts, axis=0)
    past = split.test.past_target
    future = split.test.future_target
    naive = naive_last_value_forecast(past, cfg.prediction_length)
    return past, future, naive, pred


def select_window(
    truth: np.ndarray,
    zero_pred: np.ndarray,
    fine_pred: np.ndarray,
    requested_index: int | None,
) -> int:
    if requested_index is not None:
        if requested_index < 0 or requested_index >= truth.shape[0]:
            raise IndexError(f"window_index={requested_index} out of range 0..{truth.shape[0] - 1}")
        return requested_index

    zero_mae = np.mean(np.abs(truth - zero_pred), axis=1)
    fine_mae = np.mean(np.abs(truth - fine_pred), axis=1)
    improvement = zero_mae - fine_mae
    positive = np.where(improvement > 0)[0]
    if positive.size == 0:
        return int(np.argmin(fine_mae))
    ranked = positive[np.argsort(improvement[positive])]
    return int(ranked[len(ranked) // 2])


def plot_comparison(
    past: np.ndarray,
    truth: np.ndarray,
    naive: np.ndarray,
    zero_pred: np.ndarray,
    fine_pred: np.ndarray,
    index: int,
    output_path: Path,
    context_length: int,
) -> None:
    history = past[index, -context_length:]
    y_true = truth[index]
    y_naive = naive[index]
    y_zero = zero_pred[index]
    y_fine = fine_pred[index]

    zero_mae = float(np.mean(np.abs(y_true - y_zero)))
    fine_mae = float(np.mean(np.abs(y_true - y_fine)))
    naive_mae = float(np.mean(np.abs(y_true - y_naive)))

    x_hist = np.arange(len(history))
    x_future = np.arange(len(history), len(history) + len(y_true))

    plt.figure(figsize=(10, 4.8))
    plt.plot(x_hist, history, color="#4b5563", linewidth=1.8, label="history")
    plt.plot(x_future, y_true, color="#111827", linewidth=2.4, label="ground truth")
    plt.plot(x_future, y_naive, color="#9ca3af", linewidth=1.8, linestyle="--", label=f"naive MAE={naive_mae:.2f}")
    plt.plot(x_future, y_zero, color="#2563eb", linewidth=2.0, label=f"zero-shot MAE={zero_mae:.2f}")
    plt.plot(x_future, y_fine, color="#dc2626", linewidth=2.0, label=f"fine-tuned MAE={fine_mae:.2f}")
    plt.axvline(len(history) - 1, color="#6b7280", linewidth=1.0, alpha=0.5)
    plt.title(f"ETTm2 Forecast Comparison, Window {index}")
    plt.xlabel("Time step")
    plt.ylabel("OT")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    print(f"saved {output_path}")


def plot_error_comparison(
    truth: np.ndarray,
    zero_pred: np.ndarray,
    fine_pred: np.ndarray,
    index: int,
    output_path: Path,
) -> None:
    zero_abs_error = np.abs(truth[index] - zero_pred[index])
    fine_abs_error = np.abs(truth[index] - fine_pred[index])
    x = np.arange(len(zero_abs_error))

    plt.figure(figsize=(8, 3.8))
    plt.plot(x, zero_abs_error, marker="o", linewidth=1.8, label="zero-shot absolute error")
    plt.plot(x, fine_abs_error, marker="s", linewidth=1.8, label="fine-tuned absolute error")
    plt.title(f"Absolute Error Comparison, Window {index}")
    plt.xlabel("Forecast horizon")
    plt.ylabel("Absolute error")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    print(f"saved {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device_target", default="CPU")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--window_stride", type=int, default=96)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--window_index", type=int, default=None)
    parser.add_argument("--zero_checkpoint", default="results/checkpoints/lag_llama_ett_pretrain_best.ckpt")
    parser.add_argument("--finetune_checkpoint", default="results/checkpoints/lag_llama_ettm2_finetune_best.ckpt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    device = configure_device(args.device_target)

    zero_cfg = ett_zero_shot_config()
    zero_cfg.device_target = device
    zero_cfg.batch_size = args.batch_size
    zero_cfg.window_stride = args.window_stride
    zero_cfg.log_every = args.log_every

    fine_cfg = ett_finetune_config()
    fine_cfg.device_target = device
    fine_cfg.batch_size = args.batch_size
    fine_cfg.window_stride = args.window_stride
    fine_cfg.log_every = args.log_every

    past, truth, naive, zero_pred = predict_all(zero_cfg, args.zero_checkpoint)
    _, fine_truth, _, fine_pred = predict_all(fine_cfg, args.finetune_checkpoint)
    if not np.allclose(truth, fine_truth):
        raise ValueError("Zero-shot and fine-tuned evaluation windows do not match.")

    index = select_window(truth, zero_pred, fine_pred, args.window_index)
    zero_mae = float(np.mean(np.abs(truth[index] - zero_pred[index])))
    fine_mae = float(np.mean(np.abs(truth[index] - fine_pred[index])))
    print(f"selected window={index} zero_shot_mae={zero_mae:.4f} fine_tuned_mae={fine_mae:.4f}")

    plot_comparison(
        past,
        truth,
        naive,
        zero_pred,
        fine_pred,
        index,
        FIGURES_DIR / "forecast_zero_vs_finetuned.png",
        zero_cfg.context_length,
    )
    plot_error_comparison(
        truth,
        zero_pred,
        fine_pred,
        index,
        FIGURES_DIR / "forecast_error_zero_vs_finetuned.png",
    )


if __name__ == "__main__":
    main()
