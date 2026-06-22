from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
DOWNSTREAMS = ("ettm2", "exchange_rate", "weather")


def load_metrics(path: Path) -> dict | None:
    if not path.exists():
        print(f"skip missing metrics: {path}")
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["metrics"]


def metric_value(metrics: dict, key: str) -> float:
    value = metrics.get(key)
    if value is None:
        raise KeyError(f"Missing metric {key!r}")
    return float(value)


def load_paper_baseline(dataset: str, metric_name: str) -> tuple[str, float] | None:
    candidates: list[tuple[str, float]] = []

    stat_path = RESULTS_DIR / f"paper_baselines_{dataset}.json"
    if stat_path.exists():
        with open(stat_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        best_name = payload.get("best_available_paper_baseline") or ""
        value = payload.get(f"best_available_paper_baseline_{metric_name}")
        if best_name and value is not None:
            candidates.append((best_name, float(value)))

    deep_path = RESULTS_DIR / f"paper_deep_baselines_{dataset}.json"
    if deep_path.exists():
        with open(deep_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        best_name = payload.get("best_available_deep_baseline") or ""
        value = payload.get(f"best_available_deep_baseline_{metric_name}")
        if best_name and value is not None:
            candidates.append((best_name, float(value)))

    if not candidates:
        return None
    return min(candidates, key=lambda item: item[1])


def plot_metric(metric_name: str, output_path: Path) -> None:
    labels = []
    best_baseline_labels = []
    best_baseline_values = []
    zero_shot = []
    fine_tuned = []

    for dataset in DOWNSTREAMS:
        zero = load_metrics(RESULTS_DIR / f"metrics_paper10_zero_shot_{dataset}.json")
        fine = load_metrics(RESULTS_DIR / f"metrics_paper10_finetuned_{dataset}.json")
        if zero is None or fine is None:
            continue

        paper_baseline = load_paper_baseline(dataset, metric_name)
        if paper_baseline is None:
            baseline_label = str(zero.get("best_classical_baseline", "classical"))
            baseline_value = metric_value(zero, f"best_classical_baseline_{metric_name}")
        else:
            baseline_label, baseline_value = paper_baseline

        labels.append(dataset)
        best_baseline_labels.append(baseline_label)
        best_baseline_values.append(baseline_value)
        zero_shot.append(metric_value(zero, f"model_{metric_name}"))
        fine_tuned.append(metric_value(fine, f"model_{metric_name}"))

    if not labels:
        print(f"skip {metric_name} summary: no complete metrics found")
        return

    x = np.arange(len(labels))
    width = 0.26
    plt.figure(figsize=(9, 4.6))
    bars = [
        plt.bar(x - width, best_baseline_values, width, label="Best available paper/classical baseline"),
        plt.bar(x, zero_shot, width, label="Lag-Llama zero-shot"),
        plt.bar(x + width, fine_tuned, width, label="Lag-Llama fine-tuned"),
    ]
    plt.xticks(x, labels)
    plt.ylabel(metric_name.upper())
    plt.title(f"Paper10 Downstream {metric_name.upper()} Comparison")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    for group in bars:
        for bar in group:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    for idx, label in enumerate(best_baseline_labels):
        plt.text(
            x[idx] - width,
            best_baseline_values[idx],
            label,
            ha="center",
            va="top",
            rotation=90,
            fontsize=7,
            color="#374151",
        )
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    print(f"saved {output_path}")


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_metric("mae", FIGURES_DIR / "paper10_downstream_mae_comparison.png")
    plot_metric("mse", FIGURES_DIR / "paper10_downstream_mse_comparison.png")


if __name__ == "__main__":
    main()
