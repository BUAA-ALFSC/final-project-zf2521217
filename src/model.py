from __future__ import annotations

import math
from dataclasses import dataclass

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor, Parameter


@dataclass
class LagLlamaConfig:
    context_length: int
    prediction_length: int
    lags_seq: tuple[int, ...]
    n_layer: int
    n_head: int
    n_embd_per_head: int
    scaling: str = "robust"
    use_time_features: bool = True
    num_time_features: int = 6
    dropout: float = 0.0
    loss: str = "student_t"
    student_t_fixed_df: bool = False
    eps: float = 1e-5

    @property
    def d_model(self) -> int:
        return self.n_head * self.n_embd_per_head

    @property
    def max_lag(self) -> int:
        return max(self.lags_seq) if self.lags_seq else 0

    @property
    def feature_size(self) -> int:
        time_size = self.num_time_features if self.use_time_features else 0
        return len(self.lags_seq) + 2 + time_size

    @property
    def output_dim(self) -> int:
        if self.loss != "student_t":
            return 1
        return 2 if self.student_t_fixed_df else 3


class RMSNorm(nn.Cell):
    def __init__(self, size: int, eps: float = 1e-5):
        super().__init__()
        self.scale = Parameter(ops.ones((size,), ms.float32))
        self.eps = eps

    def construct(self, x: Tensor) -> Tensor:
        norm_x = ops.mean(ops.square(x.astype(ms.float32)), axis=-1, keep_dims=True)
        x_normed = x * ops.rsqrt(norm_x + self.eps)
        return x_normed * self.scale


class RotaryEmbedding(nn.Cell):
    def __init__(self, dim: int, max_seq_len: int = 4096, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (ops.arange(0, dim, 2).astype(ms.float32) / dim))
        t = ops.arange(0, max_seq_len).astype(ms.float32)
        freqs = ops.outer(t, inv_freq)
        emb = ops.concat((freqs, freqs), axis=-1)
        self.cos_cached = ops.cos(emb).reshape(1, 1, max_seq_len, dim)
        self.sin_cached = ops.sin(emb).reshape(1, 1, max_seq_len, dim)

    def construct(self, seq_len: int) -> tuple[Tensor, Tensor]:
        return self.cos_cached[:, :, :seq_len, :], self.sin_cached[:, :, :seq_len, :]


def rotate_half(x: Tensor) -> Tensor:
    dim = x.shape[-1]
    x1 = x[..., : dim // 2]
    x2 = x[..., dim // 2 :]
    return ops.concat((-x2, x1), axis=-1)


def apply_rotary_pos_emb(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor]:
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def softplus(x: Tensor) -> Tensor:
    # MindSpore 2.0.0rc1 does not expose ops.softplus on CPU.
    # Stable formula: log(1 + exp(x)) = max(x, 0) + log(1 + exp(-abs(x))).
    return ops.maximum(x, ops.zeros_like(x)) + ops.log(ops.exp(-ops.abs(x)) + 1.0)


class CausalSelfAttention(nn.Cell):
    def __init__(self, config: LagLlamaConfig):
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.n_embd_per_head
        self.d_model = config.d_model
        self.q_proj = nn.Dense(self.d_model, self.d_model, has_bias=False)
        self.kv_proj = nn.Dense(self.d_model, 2 * self.d_model, has_bias=False)
        self.c_proj = nn.Dense(self.d_model, self.d_model, has_bias=False)
        self.rotary_emb = RotaryEmbedding(self.head_dim)
        self.dropout = nn.Dropout(p=config.dropout)

    def construct(self, x: Tensor) -> Tensor:
        batch, seq_len, _ = x.shape
        q = self.q_proj(x)
        kv = self.kv_proj(x)
        k, v = ops.split(kv, self.d_model, axis=-1)

        q = q.reshape(batch, seq_len, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch, seq_len, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch, seq_len, self.n_head, self.head_dim).transpose(0, 2, 1, 3)

        cos, sin = self.rotary_emb(seq_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        scores = ops.matmul(q, k.swapaxes(-1, -2)) / math.sqrt(self.head_dim)
        mask = ops.tril(ops.ones((seq_len, seq_len), ms.float32)).reshape(1, 1, seq_len, seq_len)
        scores = ops.where(mask > 0, scores, ops.full_like(scores, -1.0e9))
        attn = ops.softmax(scores, axis=-1)
        attn = self.dropout(attn)
        y = ops.matmul(attn, v)
        y = y.transpose(0, 2, 1, 3).reshape(batch, seq_len, self.d_model)
        return self.c_proj(y)


def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)


class MLP(nn.Cell):
    def __init__(self, config: LagLlamaConfig):
        super().__init__()
        hidden_dim = 4 * config.d_model
        n_hidden = find_multiple(int(2 * hidden_dim / 3), 256)
        self.c_fc1 = nn.Dense(config.d_model, n_hidden, has_bias=False)
        self.c_fc2 = nn.Dense(config.d_model, n_hidden, has_bias=False)
        self.c_proj = nn.Dense(n_hidden, config.d_model, has_bias=False)

    def construct(self, x: Tensor) -> Tensor:
        return self.c_proj(ops.silu(self.c_fc1(x)) * self.c_fc2(x))


class TransformerBlock(nn.Cell):
    def __init__(self, config: LagLlamaConfig):
        super().__init__()
        self.rms_1 = RMSNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.rms_2 = RMSNorm(config.d_model)
        self.mlp = MLP(config)

    def construct(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.rms_1(x))
        return x + self.mlp(self.rms_2(x))


class LagLlamaMindSpore(nn.Cell):
    def __init__(self, config: LagLlamaConfig):
        super().__init__()
        self.config = config
        self.wte = nn.Dense(config.feature_size, config.d_model)
        self.blocks = nn.CellList([TransformerBlock(config) for _ in range(config.n_layer)])
        self.ln_f = RMSNorm(config.d_model)
        self.param_proj = nn.Dense(config.d_model, config.output_dim)
        self.sort = ops.Sort(axis=1)

    def _scale(self, past_target: Tensor, observed: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        del observed
        if self.config.scaling == "robust":
            sorted_target = self.sort(past_target)[0]
            length = past_target.shape[1]
            median_idx = length // 2
            q25_idx = max(0, int((length - 1) * 0.25))
            q75_idx = max(0, int((length - 1) * 0.75))
            loc = sorted_target[:, median_idx : median_idx + 1]
            q25 = sorted_target[:, q25_idx : q25_idx + 1]
            q75 = sorted_target[:, q75_idx : q75_idx + 1]
            scale = ops.maximum(q75 - q25, Tensor(1e-5, ms.float32))
        elif self.config.scaling == "mean":
            loc = ops.mean(past_target, axis=1, keep_dims=True)
            scale = ops.maximum(ops.mean(ops.abs(past_target - loc), axis=1, keep_dims=True), Tensor(1e-5, ms.float32))
        elif self.config.scaling == "std":
            loc = ops.mean(past_target, axis=1, keep_dims=True)
            scale = ops.maximum(ops.sqrt(ops.mean(ops.square(past_target - loc), axis=1, keep_dims=True)), Tensor(1e-5, ms.float32))
        else:
            loc = ops.zeros((past_target.shape[0], 1), ms.float32)
            scale = ops.ones((past_target.shape[0], 1), ms.float32)
        return (past_target - loc) / scale, loc, scale

    def prepare_input(
        self,
        past_target: Tensor,
        past_observed_values: Tensor,
        past_time_feat: Tensor | None = None,
        future_time_feat: Tensor | None = None,
        future_target: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        scaled_past_target, loc, scale = self._scale(past_target, past_observed_values)
        max_lag = self.config.max_lag

        if future_target is not None:
            future_teacher = (future_target[:, :-1] - loc) / scale
            input_series = ops.concat((scaled_past_target[:, max_lag:], future_teacher), axis=1)
        else:
            input_series = scaled_past_target[:, max_lag:]

        features = []
        total_len = input_series.shape[1]
        full_series = ops.concat((scaled_past_target[:, :max_lag], input_series), axis=1) if max_lag > 0 else input_series
        for lag in self.config.lags_seq:
            start = max_lag - lag
            features.append(full_series[:, start : start + total_len].expand_dims(-1))

        static_feat = ops.concat((ops.log1p(ops.abs(loc)), ops.log(scale)), axis=1)
        static_feat = static_feat.expand_dims(1)
        static_feat = ops.tile(static_feat, (1, total_len, 1))
        if self.config.use_time_features and past_time_feat is not None:
            if future_time_feat is not None and future_target is not None:
                time_feat = ops.concat(
                    (past_time_feat[:, max_lag:, :], future_time_feat[:, :-1, :]),
                    axis=1,
                )
            else:
                time_feat = past_time_feat[:, max_lag:, :]
            return ops.concat(features + [static_feat, time_feat], axis=-1), loc, scale
        return ops.concat(features + [static_feat], axis=-1), loc, scale

    def construct(
        self,
        past_target: Tensor,
        past_observed_values: Tensor,
        past_time_feat: Tensor | None = None,
        future_time_feat: Tensor | None = None,
        future_target: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        transformer_input, loc, scale = self.prepare_input(
            past_target=past_target,
            past_observed_values=past_observed_values,
            past_time_feat=past_time_feat,
            future_time_feat=future_time_feat,
            future_target=future_target,
        )
        x = self.wte(transformer_input)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        params = self.param_proj(x)
        return params, loc, scale

    def predict_mean(
        self,
        past_target: Tensor,
        past_observed_values: Tensor,
        prediction_length: int,
        past_time_feat: Tensor | None = None,
        future_time_feat: Tensor | None = None,
    ) -> Tensor:
        outputs = []
        current_past = past_target
        current_observed = past_observed_values
        current_past_time_feat = past_time_feat
        for step in range(prediction_length):
            params, loc, scale = self.construct(
                current_past,
                current_observed,
                past_time_feat=current_past_time_feat,
                future_target=None,
            )
            last_params = params[:, -1, :]
            if self.config.loss == "student_t":
                next_scaled = last_params[:, 0:1]
            else:
                next_scaled = last_params[:, 0:1]
            next_value = next_scaled * scale + loc
            outputs.append(next_value)
            current_past = ops.concat((current_past[:, 1:], next_value), axis=1)
            current_observed = ops.concat((current_observed[:, 1:], ops.ones_like(next_value)), axis=1)
            if current_past_time_feat is not None and future_time_feat is not None:
                next_time_feat = future_time_feat[:, step : step + 1, :]
                current_past_time_feat = ops.concat((current_past_time_feat[:, 1:, :], next_time_feat), axis=1)
        return ops.concat(outputs, axis=1)

    def predict_samples(
        self,
        past_target: Tensor,
        past_observed_values: Tensor,
        prediction_length: int,
        num_samples: int,
        past_time_feat: Tensor | None = None,
        future_time_feat: Tensor | None = None,
    ) -> Tensor:
        if self.config.loss != "student_t":
            mean = self.predict_mean(
                past_target,
                past_observed_values,
                prediction_length,
                past_time_feat=past_time_feat,
                future_time_feat=future_time_feat,
            )
            return ops.tile(mean.expand_dims(1), (1, num_samples, 1))

        all_paths = []
        if num_samples <= 1:
            sample_offsets = Tensor([0.0], ms.float32)
        else:
            sample_offsets = Tensor(
                [(-1.0 + 2.0 * i / (num_samples - 1)) for i in range(num_samples)],
                ms.float32,
            )
        for sample_idx in range(num_samples):
            outputs = []
            current_past = past_target
            current_observed = past_observed_values
            current_past_time_feat = past_time_feat
            for step in range(prediction_length):
                params, loc, scale = self.construct(
                    current_past,
                    current_observed,
                    past_time_feat=current_past_time_feat,
                    future_target=None,
                )
                last_params = params[:, -1, :]
                pred_loc = last_params[:, 0:1]
                pred_scale = softplus(last_params[:, 1:2]) + 1e-5
                if self.config.student_t_fixed_df:
                    pred_df = ops.ones_like(pred_scale) * 5.0
                else:
                    pred_df = softplus(last_params[:, 2:3]) + 2.0
                # Use a deterministic normal quantile grid and Student-T scale
                # correction. This keeps sampling backend-stable while reflecting
                # the learned heavy-tail parameter.
                offset = sample_offsets[sample_idx].reshape(1, 1)
                df_scale = ops.sqrt(pred_df / ops.maximum(pred_df - 2.0, Tensor(1e-5, ms.float32)))
                next_scaled = pred_loc + pred_scale * df_scale * offset
                next_value = next_scaled * scale + loc
                outputs.append(next_value)
                current_past = ops.concat((current_past[:, 1:], next_value), axis=1)
                current_observed = ops.concat((current_observed[:, 1:], ops.ones_like(next_value)), axis=1)
                if current_past_time_feat is not None and future_time_feat is not None:
                    next_time_feat = future_time_feat[:, step : step + 1, :]
                    current_past_time_feat = ops.concat((current_past_time_feat[:, 1:, :], next_time_feat), axis=1)
            all_paths.append(ops.concat(outputs, axis=1).expand_dims(1))
        return ops.concat(all_paths, axis=1)


def student_t_nll(
    target: Tensor,
    raw_params: Tensor,
    loc: Tensor,
    scale: Tensor,
    fixed_df: bool = False,
) -> Tensor:
    pred_loc = raw_params[..., 0]
    pred_scale = softplus(raw_params[..., 1]) + 1e-5
    if fixed_df:
        pred_df = Tensor(5.0, ms.float32)
        log_gamma_ratio = Tensor(
            math.lgamma((5.0 + 1.0) / 2.0) - math.lgamma(5.0 / 2.0),
            ms.float32,
        )
        log_norm = (
            log_gamma_ratio
            - 0.5 * math.log(5.0 * math.pi)
            - ops.log(pred_scale)
        )
    else:
        pred_df = softplus(raw_params[..., 2]) + 2.0
        log_norm = (
            ops.lgamma((pred_df + 1.0) / 2.0)
            - ops.lgamma(pred_df / 2.0)
            - 0.5 * ops.log(pred_df * math.pi)
            - ops.log(pred_scale)
        )
    target_scaled = (target - loc) / scale
    z = (target_scaled - pred_loc) / pred_scale
    log_prob = log_norm - ((pred_df + 1.0) / 2.0) * ops.log1p(ops.square(z) / pred_df)
    return -log_prob


class LagLlamaLoss(nn.Cell):
    def __init__(self, model: LagLlamaMindSpore):
        super().__init__(auto_prefix=False)
        self.model = model

    def construct(
        self,
        past_target: Tensor,
        past_observed_values: Tensor,
        past_time_feat: Tensor,
        future_time_feat: Tensor,
        future_target: Tensor,
        future_observed_values: Tensor,
    ) -> Tensor:
        params, loc, scale = self.model(
            past_target,
            past_observed_values,
            past_time_feat=past_time_feat,
            future_time_feat=future_time_feat,
            future_target=future_target,
        )
        context_target = past_target[:, -(self.model.config.context_length - 1) :]
        target = ops.concat((context_target, future_target), axis=1)
        context_observed = past_observed_values[:, -(self.model.config.context_length - 1) :]
        observed = ops.concat((context_observed, future_observed_values), axis=1)

        if self.model.config.loss == "student_t":
            loss = student_t_nll(
                target,
                params,
                loc,
                scale,
                fixed_df=self.model.config.student_t_fixed_df,
            )
        else:
            pred = params[..., 0] * scale + loc
            loss = ops.square(pred - target)
        return ops.sum(loss * observed) / ops.maximum(ops.sum(observed), Tensor(1.0, ms.float32))
