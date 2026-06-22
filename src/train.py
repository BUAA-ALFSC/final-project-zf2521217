from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

try:
    import mindspore as ms
    import mindspore.nn as nn
    import mindspore.ops as ops
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
    ett_quick_config,
    ett_finetune_config,
    ett_pretrain_config,
    paper10_finetune_config,
    paper10_pretrain_config,
    smoke_config,
)
from .data import iter_batches, load_experiment_data
from .device import configure_device
from .model import LagLlamaConfig, LagLlamaLoss, LagLlamaMindSpore


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    ms.set_seed(seed)


def build_model_config(cfg: ExperimentConfig) -> LagLlamaConfig:
    return LagLlamaConfig(
        context_length=cfg.context_length,
        prediction_length=cfg.prediction_length,
        lags_seq=cfg.lags_seq,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        n_embd_per_head=cfg.n_embd_per_head,
        scaling=cfg.scaling,
        use_time_features=cfg.use_time_features,
        dropout=cfg.dropout,
        loss=cfg.loss,
        student_t_fixed_df=cfg.student_t_fixed_df,
    )


def time_feature_tensor(value: np.ndarray | None, batch_size: int, length: int) -> Tensor:
    if value is None:
        return Tensor(np.zeros((batch_size, length, 6), dtype=np.float32), ms.float32)
    return Tensor(value, ms.float32)


def write_history(cfg: ExperimentConfig, history: dict) -> None:
    with open(RESULTS_DIR / cfg.history_file, "w", encoding="utf-8") as f:
        json.dump({"config": cfg.to_dict(), "history": history}, f, indent=2)
    with open(RESULTS_DIR / "train_history.json", "w", encoding="utf-8") as f:
        json.dump({"config": cfg.to_dict(), "history": history}, f, indent=2)


def frequency_mask(
    past_target: np.ndarray,
    future_target: np.ndarray,
    rate: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if rate <= 0:
        return past_target, future_target
    past_len = past_target.shape[1]
    future_len = future_target.shape[1]
    window = np.concatenate([past_target, future_target], axis=1)
    window_freq = np.fft.rfft(window, axis=1)
    mask = rng.random(window_freq.shape) < rate
    window_freq = np.where(mask, 0.0 + 0.0j, window_freq)
    augmented = np.fft.irfft(window_freq, n=past_len + future_len, axis=1)
    return (
        augmented[:, :past_len].astype(np.float32),
        augmented[:, past_len:].astype(np.float32),
    )


def frequency_mix(
    past_target: np.ndarray,
    future_target: np.ndarray,
    rate: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if rate <= 0 or past_target.shape[0] < 2:
        return past_target, future_target
    past_len = past_target.shape[1]
    future_len = future_target.shape[1]
    window = np.concatenate([past_target, future_target], axis=1)
    window_freq = np.fft.rfft(window, axis=1)
    mask = rng.random(window_freq.shape) < rate

    amplitude = np.abs(window_freq)
    dominant_order = np.argsort(-amplitude, axis=1)
    dominant_rank = np.empty_like(dominant_order)
    rank_values = np.arange(dominant_order.shape[1])[None, :]
    np.put_along_axis(dominant_rank, dominant_order, rank_values, axis=1)
    mask = np.logical_and(mask, dominant_rank > 2)

    shuffled = rng.permutation(past_target.shape[0])
    mixed_window = np.concatenate([past_target[shuffled], future_target[shuffled]], axis=1)
    mixed_freq = np.fft.rfft(mixed_window, axis=1)
    combined_freq = np.where(mask, mixed_freq, window_freq)
    augmented = np.fft.irfft(combined_freq, n=past_len + future_len, axis=1)
    return (
        augmented[:, :past_len].astype(np.float32),
        augmented[:, past_len:].astype(np.float32),
    )


def augment_batch(batch: dict, cfg: ExperimentConfig, rng: np.random.Generator) -> dict:
    if cfg.aug_prob <= 0 or rng.random() >= cfg.aug_prob:
        return batch
    past_target = batch["past_target"].copy()
    future_target = batch["future_target"].copy()
    past_target, future_target = frequency_mask(
        past_target,
        future_target,
        cfg.freq_mask_rate,
        rng,
    )
    past_target, future_target = frequency_mix(
        past_target,
        future_target,
        cfg.freq_mixing_rate,
        rng,
    )
    augmented = dict(batch)
    augmented["past_target"] = past_target
    augmented["future_target"] = future_target
    return augmented


def run_training(cfg: ExperimentConfig) -> dict:
    os.environ.setdefault("MPLCONFIGDIR", str(RESULTS_DIR / ".matplotlib"))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    cfg.device_target = configure_device(cfg.device_target)
    set_seed(cfg.seed)

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
    if cfg.checkpoint_path:
        if not Path(cfg.checkpoint_path).exists():
            raise FileNotFoundError(f"checkpoint_path not found: {cfg.checkpoint_path}")
        ms.load_checkpoint(cfg.checkpoint_path, net=model)
        print(f"Loaded checkpoint: {cfg.checkpoint_path}")
    loss_net = LagLlamaLoss(model)
    optimizer = nn.Adam(
        model.trainable_params(),
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    grad_fn = ms.value_and_grad(loss_net, None, optimizer.parameters)
    augment_rng = np.random.default_rng(cfg.seed + 1729)

    def train_step(batch: dict) -> Tensor:
        past_target = Tensor(batch["past_target"], ms.float32)
        past_observed_values = Tensor(batch["past_observed_values"], ms.float32)
        future_target = Tensor(batch["future_target"], ms.float32)
        future_observed_values = Tensor(batch["future_observed_values"], ms.float32)
        past_time_feat = time_feature_tensor(batch["past_time_feat"], past_target.shape[0], past_target.shape[1])
        future_time_feat = time_feature_tensor(batch["future_time_feat"], future_target.shape[0], future_target.shape[1])
        loss, grads = grad_fn(
            past_target,
            past_observed_values,
            past_time_feat,
            future_time_feat,
            future_target,
            future_observed_values,
        )
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
    epochs_without_improvement = 0
    stem = Path(cfg.output_checkpoint).stem
    for epoch in range(1, cfg.epochs + 1):
        model.set_train(True)
        train_losses = []
        for batch_idx, batch in enumerate(iter_batches(split.train, cfg.batch_size, shuffle=True, seed=cfg.seed + epoch), start=1):
            batch = augment_batch(batch, cfg, augment_rng)
            batch_loss = float(train_step(batch).asnumpy())
            train_losses.append(batch_loss)
            if cfg.log_every and batch_idx % cfg.log_every == 0:
                print(f"epoch={epoch:03d} batch={batch_idx:05d} train_loss={batch_loss:.6f}", flush=True)
            if cfg.max_train_batches and batch_idx >= cfg.max_train_batches:
                break

        model.set_train(False)
        val_losses = []
        for val_batch_idx, batch in enumerate(iter_batches(split.val, cfg.batch_size, shuffle=False, seed=cfg.seed), start=1):
            loss = loss_net(
                Tensor(batch["past_target"], ms.float32),
                Tensor(batch["past_observed_values"], ms.float32),
                time_feature_tensor(batch["past_time_feat"], batch["past_target"].shape[0], batch["past_target"].shape[1]),
                time_feature_tensor(batch["future_time_feat"], batch["future_target"].shape[0], batch["future_target"].shape[1]),
                Tensor(batch["future_target"], ms.float32),
                Tensor(batch["future_observed_values"], ms.float32),
            )
            val_losses.append(float(loss.asnumpy()))
            if cfg.max_val_batches and val_batch_idx >= cfg.max_val_batches:
                break

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        history["best_val_loss"].append(best_val_loss)
        history["best_epoch"] = best_epoch
        print(
            f"epoch={epoch:03d} train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f} best_val_loss={best_val_loss:.6f} "
            f"best_epoch={best_epoch:03d}"
        )
        epoch_ckpt_path = CHECKPOINT_DIR / f"{stem}_epoch_{epoch:03d}.ckpt"
        latest_ckpt_path = CHECKPOINT_DIR / f"{stem}_latest.ckpt"
        best_ckpt_path = CHECKPOINT_DIR / f"{stem}_best.ckpt"
        ms.save_checkpoint(model, str(epoch_ckpt_path))
        ms.save_checkpoint(model, str(latest_ckpt_path))
        if improved:
            ms.save_checkpoint(model, str(best_ckpt_path))
        write_history(cfg, history)
        if cfg.early_stopping_patience and epochs_without_improvement >= cfg.early_stopping_patience:
            history["early_stopped"] = True
            history["stopped_epoch"] = epoch
            print(
                f"early stopping: no val_loss improvement for "
                f"{cfg.early_stopping_patience} epochs; best_epoch={best_epoch:03d}"
            )
            write_history(cfg, history)
            break

    ckpt_path = CHECKPOINT_DIR / cfg.output_checkpoint
    ms.save_checkpoint(model, str(ckpt_path))
    write_history(cfg, history)
    return {"checkpoint": str(ckpt_path), "history": history, "config": cfg.to_dict()}


def parse_args(argv: list[str] | None = None) -> ExperimentConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Run tiny synthetic smoke test only.")
    parser.add_argument("--ett_pretrain", action="store_true", help="Use ETT-H1/H2/M1 pretraining config.")
    parser.add_argument("--ett_quick", action="store_true", help="Use a fast ETT flow check for CPU debugging.")
    parser.add_argument("--ett_finetune", action="store_true", help="Use ETT-M2 finetuning config.")
    parser.add_argument("--paper10_pretrain", action="store_true", help="Use the paper-dataset-subset pretraining config.")
    parser.add_argument("--paper10_finetune", action="store_true", help="Use the default ETT-M2 finetuning config for paper10 checkpoint.")
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
    parser.add_argument("--history_file", default="")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--context_length", type=int, default=None)
    parser.add_argument("--prediction_length", type=int, default=None)
    parser.add_argument("--loss", choices=["student_t", "mse"], default=None)
    parser.add_argument("--student_t_fixed_df", action="store_true")
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--device_target", default="")
    parser.add_argument("--window_stride", type=int, default=None)
    parser.add_argument("--max_train_batches", type=int, default=None)
    parser.add_argument("--max_val_batches", type=int, default=None)
    parser.add_argument("--log_every", type=int, default=None)
    parser.add_argument("--early_stopping_patience", type=int, default=None)
    parser.add_argument("--aug_prob", type=float, default=None)
    parser.add_argument("--freq_mask_rate", type=float, default=None)
    parser.add_argument("--freq_mixing_rate", type=float, default=None)
    parser.add_argument("--no_augmentation", action="store_true")
    parser.add_argument("--no_time_features", action="store_true")
    args = parser.parse_args(argv)

    if args.smoke:
        cfg = smoke_config()
    elif args.ett_pretrain:
        cfg = ett_pretrain_config()
    elif args.ett_quick:
        cfg = ett_quick_config()
    elif args.ett_finetune:
        cfg = ett_finetune_config()
    elif args.paper10_pretrain:
        cfg = paper10_pretrain_config()
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
    if args.history_file:
        cfg.history_file = args.history_file
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.learning_rate is not None:
        cfg.learning_rate = args.learning_rate
    if args.weight_decay is not None:
        cfg.weight_decay = args.weight_decay
    if args.context_length is not None:
        cfg.context_length = args.context_length
    if args.prediction_length is not None:
        cfg.prediction_length = args.prediction_length
    if args.loss is not None:
        cfg.loss = args.loss
    if args.student_t_fixed_df:
        cfg.student_t_fixed_df = True
    if args.num_samples is not None:
        cfg.num_samples = args.num_samples
    if args.device_target:
        cfg.device_target = args.device_target
    if args.window_stride is not None:
        cfg.window_stride = args.window_stride
    if args.max_train_batches is not None:
        cfg.max_train_batches = args.max_train_batches
    if args.max_val_batches is not None:
        cfg.max_val_batches = args.max_val_batches
    if args.log_every is not None:
        cfg.log_every = args.log_every
    if args.early_stopping_patience is not None:
        cfg.early_stopping_patience = args.early_stopping_patience
    if args.aug_prob is not None:
        cfg.aug_prob = args.aug_prob
    if args.freq_mask_rate is not None:
        cfg.freq_mask_rate = args.freq_mask_rate
    if args.freq_mixing_rate is not None:
        cfg.freq_mixing_rate = args.freq_mixing_rate
    if args.no_augmentation:
        cfg.aug_prob = 0.0
        cfg.freq_mask_rate = 0.0
        cfg.freq_mixing_rate = 0.0
    if args.no_time_features:
        cfg.use_time_features = False
    return cfg


def main(argv: list[str] | None = None) -> None:
    cfg = parse_args(argv)
    result = run_training(cfg)
    print(json.dumps(result["config"], indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])
