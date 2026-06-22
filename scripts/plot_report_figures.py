from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_history(history_path: Path, output_path: Path, title: str) -> None:
    if not history_path.exists():
        print(f"skip missing history: {history_path}")
        return

    payload = load_json(history_path)
    history = payload["history"]
    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])
    epochs = np.arange(1, max(len(train_loss), len(val_loss)) + 1)

    plt.figure(figsize=(8, 4))
    if train_loss:
        plt.plot(epochs[: len(train_loss)], train_loss, marker="o", linewidth=1.5, label="train loss")
    if val_loss:
        plt.plot(epochs[: len(val_loss)], val_loss, marker="s", linewidth=1.5, label="validation loss")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Negative log-likelihood")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    print(f"saved {output_path}")


def plot_metric_bars(zero_path: Path, finetuned_path: Path, output_path: Path) -> None:
    if not zero_path.exists() or not finetuned_path.exists():
        print("skip metrics bar plot: missing zero-shot or fine-tuned metrics")
        return

    zero_metrics = load_json(zero_path)["metrics"]
    finetuned_metrics = load_json(finetuned_path)["metrics"]

    methods = ["Naive", "Zero-shot", "Fine-tuned"]
    mae_values = [
        zero_metrics["naive_mae"],
        zero_metrics["model_mae"],
        finetuned_metrics["model_mae"],
    ]
    mse_values = [
        zero_metrics["naive_mse"],
        zero_metrics["model_mse"],
        finetuned_metrics["model_mse"],
    ]

    x = np.arange(len(methods))
    width = 0.36

    plt.figure(figsize=(8, 4))
    mae_bars = plt.bar(x - width / 2, mae_values, width, label="MAE")
    mse_bars = plt.bar(x + width / 2, mse_values, width, label="MSE")
    plt.xticks(x, methods)
    plt.ylabel("Error")
    plt.title("ETTm2 Forecasting Error Comparison")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()

    for bars in (mae_bars, mse_bars):
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    print(f"saved {output_path}")


def plot_mae_reduction(zero_path: Path, finetuned_path: Path, output_path: Path) -> None:
    if not zero_path.exists() or not finetuned_path.exists():
        print("skip MAE reduction plot: missing zero-shot or fine-tuned metrics")
        return

    zero_metrics = load_json(zero_path)["metrics"]
    finetuned_metrics = load_json(finetuned_path)["metrics"]
    methods = ["Naive", "Zero-shot", "Fine-tuned"]
    mae_values = [
        zero_metrics["naive_mae"],
        zero_metrics["model_mae"],
        finetuned_metrics["model_mae"],
    ]

    plt.figure(figsize=(7, 4))
    bars = plt.bar(methods, mae_values)
    plt.ylabel("MAE")
    plt.title("MAE Reduction on ETTm2")
    plt.grid(axis="y", alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    print(f"saved {output_path}")


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plot_history(
        RESULTS_DIR / "train_history_ett_pretrain.json",
        FIGURES_DIR / "loss_ett_pretrain.png",
        "ETT Pretraining Loss",
    )
    plot_history(
        RESULTS_DIR / "train_history_ett_finetune.json",
        FIGURES_DIR / "loss_ett_finetune.png",
        "ETTm2 Fine-tuning Loss",
    )
    plot_metric_bars(
        RESULTS_DIR / "metrics_ett_zero_shot.json",
        RESULTS_DIR / "metrics_ett_finetuned.json",
        FIGURES_DIR / "metrics_comparison_ettm2.png",
    )
    plot_mae_reduction(
        RESULTS_DIR / "metrics_ett_zero_shot.json",
        RESULTS_DIR / "metrics_ett_finetuned.json",
        FIGURES_DIR / "mae_reduction_ettm2.png",
    )


if __name__ == "__main__":
    main()
