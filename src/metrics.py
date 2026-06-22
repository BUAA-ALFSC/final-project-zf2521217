from __future__ import annotations

import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def naive_last_value_forecast(past_target: np.ndarray, prediction_length: int) -> np.ndarray:
    last_value = past_target[:, -1:]
    return np.repeat(last_value, prediction_length, axis=1)


def moving_average_forecast(
    past_target: np.ndarray,
    prediction_length: int,
    window: int | None = None,
) -> np.ndarray:
    if window is None or window <= 0:
        window = min(24, past_target.shape[1])
    window = min(window, past_target.shape[1])
    values = np.mean(past_target[:, -window:], axis=1, keepdims=True)
    return np.repeat(values, prediction_length, axis=1)


def drift_forecast(
    past_target: np.ndarray,
    prediction_length: int,
    context_length: int | None = None,
) -> np.ndarray:
    if context_length is None or context_length <= 1:
        context_length = min(32, past_target.shape[1])
    context_length = min(context_length, past_target.shape[1])
    context = past_target[:, -context_length:]
    slope = (context[:, -1:] - context[:, :1]) / max(context_length - 1, 1)
    steps = np.arange(1, prediction_length + 1, dtype=np.float32).reshape(1, -1)
    return context[:, -1:] + slope * steps


def seasonal_naive_forecast(
    past_target: np.ndarray,
    prediction_length: int,
    season_length: int,
) -> np.ndarray:
    if season_length <= 0 or past_target.shape[1] < season_length:
        return naive_last_value_forecast(past_target, prediction_length)
    offsets = np.arange(prediction_length) % season_length
    return past_target[:, -season_length + offsets]


def classical_baseline_forecasts(
    past_target: np.ndarray,
    prediction_length: int,
    context_length: int,
    seasonal_periods: tuple[int, ...] = (),
) -> dict[str, np.ndarray]:
    forecasts = {
        "last_value": naive_last_value_forecast(past_target, prediction_length),
        "moving_average": moving_average_forecast(
            past_target,
            prediction_length,
            window=context_length,
        ),
        "drift": drift_forecast(
            past_target,
            prediction_length,
            context_length=context_length,
        ),
    }
    for period in seasonal_periods:
        if period <= 0 or period > past_target.shape[1]:
            continue
        forecasts[f"seasonal_naive_{period}"] = seasonal_naive_forecast(
            past_target,
            prediction_length,
            period,
        )
    return forecasts


def sample_crps(y_true: np.ndarray, samples: np.ndarray) -> float:
    """Sample approximation of CRPS.

    Args:
        y_true: shape [batch, prediction_length].
        samples: shape [batch, num_samples, prediction_length].
    """
    y_true = y_true[:, None, :]
    term_1 = np.mean(np.abs(samples - y_true), axis=1)
    sorted_samples = np.sort(samples, axis=1)
    n = samples.shape[1]
    weights = (2 * np.arange(1, n + 1) - n - 1).reshape(1, n, 1)
    term_2 = np.sum(weights * sorted_samples, axis=1) / (n * n)
    return float(np.mean(term_1 - term_2))


def quantile_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> float:
    diff = y_true - y_pred
    return float(2.0 * np.sum(np.maximum(quantile * diff, (quantile - 1.0) * diff)))


def mean_w_quantile_loss(
    y_true: np.ndarray,
    samples: np.ndarray,
    quantiles: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
) -> float:
    """GluonTS-style mean weighted quantile loss used as CRPS proxy.

    Lag-Llama reports GluonTS `mean_wQuantileLoss` as CRPS. We compute the same
    style of metric from forecast samples and standard quantile levels.
    """
    denominator = float(np.sum(np.abs(y_true)))
    denominator = denominator if denominator > 1e-8 else 1.0
    losses = []
    for quantile in quantiles:
        q_pred = np.quantile(samples, quantile, axis=1)
        losses.append(quantile_loss(y_true, q_pred, quantile) / denominator)
    return float(np.mean(losses))
