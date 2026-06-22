from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"


@dataclass
class ExperimentConfig:
    # Official reproduction config in lag-llama/configs/lag_llama.json uses:
    # context_length=32, n_layer=8, n_head=9, n_embd_per_head=16, robust scaling.
    # Keep the same defaults here; use smaller values only for local smoke tests.
    dataset: str = "synthetic"
    data_path: str = ""
    train_data_paths: tuple[str, ...] = ()
    val_data_path: str = ""
    test_data_path: str = ""
    target_column: str = ""
    time_column: str = "date"
    freq: str = "15min"
    series_index: int = 0
    checkpoint_path: str = ""
    output_checkpoint: str = "lag_llama_mindspore.ckpt"
    num_samples: int = 100
    context_length: int = 32
    prediction_length: int = 24
    # Multi-scale lag set for ETT-style hourly and 15-minute series.
    # It keeps short-term lags and neighbor lags around daily/weekly periods:
    # hourly: 24, 168; 15-minute: 96, 672.
    lags_seq: tuple[int, ...] = (
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        23,
        24,
        25,
        47,
        48,
        49,
        95,
        96,
        97,
        167,
        168,
        169,
        671,
        672,
        673,
    )
    n_layer: int = 8
    n_head: int = 9
    n_embd_per_head: int = 16
    dropout: float = 0.0
    scaling: Literal["robust", "mean", "std", "none"] = "robust"
    use_time_features: bool = True
    loss: Literal["student_t", "mse"] = "student_t"
    student_t_fixed_df: bool = False
    batch_size: int = 64
    epochs: int = 20
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    train_ratio: float = 0.7
    val_ratio: float = 0.1
    seed: int = 42
    num_synthetic_points: int = 5000
    device_target: str = "auto"
    window_stride: int = 1
    max_train_batches: int = 0
    max_val_batches: int = 0
    max_eval_windows: int = 0
    log_every: int = 20
    early_stopping_patience: int = 0
    aug_prob: float = 0.0
    freq_mask_rate: float = 0.0
    freq_mixing_rate: float = 0.0
    history_file: str = "train_history.json"
    metrics_file: str = "metrics.json"
    figure_file: str = "forecast_example.png"

    @property
    def d_model(self) -> int:
        return self.n_head * self.n_embd_per_head

    @property
    def max_lag(self) -> int:
        return max(self.lags_seq) if self.lags_seq else 0

    def to_dict(self) -> dict:
        data = asdict(self)
        data["d_model"] = self.d_model
        data["max_lag"] = self.max_lag
        return data


def smoke_config() -> ExperimentConfig:
    """Tiny configuration for local shape/debug runs only, not final reproduction."""
    return ExperimentConfig(
        dataset="synthetic",
        context_length=32,
        prediction_length=16,
        n_layer=2,
        n_head=4,
        n_embd_per_head=16,
        batch_size=32,
        epochs=3,
        learning_rate=3e-4,
        loss="student_t",
        scaling="mean",
        use_time_features=False,
        num_samples=10,
        num_synthetic_points=1500,
        log_every=5,
        history_file="train_history_smoke.json",
        metrics_file="metrics_smoke.json",
        figure_file="forecast_smoke.png",
    )


def ett_pretrain_config() -> ExperimentConfig:
    return ExperimentConfig(
        dataset="multi_csv",
        train_data_paths=(
            "data/ETTh1.csv",
            "data/ETTh2.csv",
            "data/ETTm1.csv",
        ),
        target_column="OT",
        time_column="date",
        freq="15min",
        context_length=32,
        prediction_length=24,
        n_layer=8,
        n_head=9,
        n_embd_per_head=16,
        loss="student_t",
        batch_size=64,
        epochs=100,
        learning_rate=1e-4,
        scaling="robust",
        use_time_features=True,
        output_checkpoint="lag_llama_ett_pretrain.ckpt",
        max_train_batches=128,
        max_val_batches=128,
        log_every=10,
        early_stopping_patience=10,
        aug_prob=0.5,
        freq_mask_rate=0.5,
        freq_mixing_rate=0.25,
        history_file="train_history_ett_pretrain.json",
    )


def paper10_pretrain_config() -> ExperimentConfig:
    """Resource-bounded multi-dataset pretraining closer to the paper setup."""
    return ExperimentConfig(
        dataset="csv_dir",
        data_path="data/paper10_pretrain",
        target_column="OT",
        time_column="date",
        freq="15min",
        context_length=32,
        prediction_length=24,
        n_layer=8,
        n_head=9,
        n_embd_per_head=16,
        loss="student_t",
        batch_size=64,
        epochs=150,
        learning_rate=1e-4,
        scaling="robust",
        use_time_features=True,
        output_checkpoint="lag_llama_paper10_pretrain.ckpt",
        window_stride=24,
        max_train_batches=256,
        max_val_batches=256,
        log_every=10,
        early_stopping_patience=15,
        aug_prob=0.5,
        freq_mask_rate=0.5,
        freq_mixing_rate=0.25,
        history_file="train_history_paper10_pretrain.json",
    )


def paper10_zero_shot_config() -> ExperimentConfig:
    return ExperimentConfig(
        dataset="csv",
        data_path="data/paper10_downstream/ETTm2.csv",
        target_column="OT",
        time_column="date",
        freq="15min",
        context_length=32,
        prediction_length=24,
        n_layer=8,
        n_head=9,
        n_embd_per_head=16,
        loss="student_t",
        batch_size=64,
        learning_rate=1e-4,
        scaling="robust",
        use_time_features=True,
        checkpoint_path="results/checkpoints/lag_llama_paper10_pretrain_best.ckpt",
        output_checkpoint="lag_llama_paper10_pretrain.ckpt",
        metrics_file="metrics_paper10_zero_shot_ettm2.json",
        figure_file="forecast_paper10_zero_shot_ettm2.png",
    )


def paper10_finetune_config() -> ExperimentConfig:
    return ExperimentConfig(
        dataset="csv",
        data_path="data/paper10_downstream/ETTm2.csv",
        target_column="OT",
        time_column="date",
        freq="15min",
        context_length=32,
        prediction_length=24,
        n_layer=8,
        n_head=9,
        n_embd_per_head=16,
        loss="student_t",
        batch_size=64,
        epochs=50,
        learning_rate=1e-5,
        scaling="robust",
        use_time_features=True,
        checkpoint_path="results/checkpoints/lag_llama_paper10_pretrain_best.ckpt",
        output_checkpoint="lag_llama_paper10_ettm2_finetune.ckpt",
        max_train_batches=256,
        max_val_batches=256,
        log_every=10,
        early_stopping_patience=10,
        aug_prob=0.5,
        freq_mask_rate=0.5,
        freq_mixing_rate=0.25,
        history_file="train_history_paper10_ettm2_finetune.json",
        metrics_file="metrics_paper10_ettm2_finetuned.json",
        figure_file="forecast_paper10_ettm2_finetuned.png",
    )


def ett_finetune_config() -> ExperimentConfig:
    return ExperimentConfig(
        dataset="csv",
        data_path="data/ETTm2.csv",
        target_column="OT",
        time_column="date",
        freq="15min",
        context_length=32,
        prediction_length=24,
        n_layer=8,
        n_head=9,
        n_embd_per_head=16,
        loss="student_t",
        batch_size=64,
        epochs=30,
        learning_rate=1e-4,
        scaling="robust",
        use_time_features=True,
        checkpoint_path="results/checkpoints/lag_llama_ett_pretrain_best.ckpt",
        output_checkpoint="lag_llama_ettm2_finetune.ckpt",
        max_train_batches=128,
        max_val_batches=128,
        log_every=10,
        early_stopping_patience=10,
        aug_prob=0.5,
        freq_mask_rate=0.5,
        freq_mixing_rate=0.25,
        history_file="train_history_ett_finetune.json",
        metrics_file="metrics_ett_finetuned.json",
        figure_file="forecast_ett_finetuned.png",
    )


def ett_zero_shot_config() -> ExperimentConfig:
    return ExperimentConfig(
        dataset="csv",
        data_path="data/ETTm2.csv",
        target_column="OT",
        time_column="date",
        freq="15min",
        context_length=32,
        prediction_length=24,
        n_layer=8,
        n_head=9,
        n_embd_per_head=16,
        loss="student_t",
        batch_size=64,
        learning_rate=1e-4,
        scaling="robust",
        use_time_features=True,
        checkpoint_path="results/checkpoints/lag_llama_ett_pretrain_best.ckpt",
        output_checkpoint="lag_llama_ett_pretrain.ckpt",
        metrics_file="metrics_ett_zero_shot.json",
        figure_file="forecast_ett_zero_shot.png",
    )


def ett_quick_config() -> ExperimentConfig:
    return ExperimentConfig(
        dataset="multi_csv",
        train_data_paths=(
            "data/ETTh1.csv",
            "data/ETTh2.csv",
            "data/ETTm1.csv",
        ),
        val_data_path="data/ETTm2.csv",
        test_data_path="data/ETTm2.csv",
        target_column="OT",
        time_column="date",
        freq="15min",
        context_length=32,
        prediction_length=24,
        n_layer=2,
        n_head=4,
        n_embd_per_head=16,
        loss="student_t",
        batch_size=32,
        epochs=1,
        learning_rate=3e-4,
        scaling="robust",
        use_time_features=True,
        output_checkpoint="lag_llama_ett_quick.ckpt",
        window_stride=24,
        max_train_batches=20,
        max_val_batches=5,
        log_every=1,
        num_samples=10,
        history_file="train_history_ett_quick.json",
        metrics_file="metrics_ett_quick.json",
        figure_file="forecast_ett_quick.png",
    )
