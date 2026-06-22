from __future__ import annotations

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor


class TimeSeriesScaler(nn.Cell):
    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def construct(self, past_target: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        loc = ops.mean(past_target, axis=1, keep_dims=True)
        scale = ops.maximum(
            ops.sqrt(ops.mean(ops.square(past_target - loc), axis=1, keep_dims=True)),
            Tensor(self.eps, ms.float32),
        )
        return (past_target - loc) / scale, loc, scale


class DeepARBaseline(nn.Cell):
    """Compact supervised DeepAR-style recurrent baseline.

    This keeps the paper baseline role, not the full GluonTS DeepAR trainer.
    """

    def __init__(self, input_size: int, hidden_size: int, prediction_length: int, num_layers: int = 2):
        super().__init__()
        self.prediction_length = prediction_length
        self.scaler = TimeSeriesScaler()
        self.input_proj = nn.Dense(input_size, hidden_size)
        self.encoder = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.head = nn.Dense(hidden_size, prediction_length)

    def construct(self, past_target: Tensor, past_time_feat: Tensor | None = None) -> Tensor:
        scaled, loc, scale = self.scaler(past_target)
        x = scaled.expand_dims(-1)
        if past_time_feat is not None:
            x = ops.concat((x, past_time_feat), axis=-1)
        x = self.input_proj(x)
        output, _ = self.encoder(x)
        hidden = output[:, -1, :]
        pred = self.head(hidden)
        return pred * scale + loc


class PatchTSTBaseline(nn.Cell):
    """Compact PatchTST-style supervised Transformer baseline."""

    def __init__(
        self,
        context_length: int,
        prediction_length: int,
        patch_length: int = 8,
        stride: int = 4,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
    ):
        super().__init__()
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.patch_length = patch_length
        self.stride = stride
        self.num_patches = max(1, (context_length - patch_length) // stride + 1)
        self.scaler = TimeSeriesScaler()
        self.patch_proj = nn.Dense(patch_length, d_model)
        self.pos_embedding = ms.Parameter(ops.zeros((1, self.num_patches, d_model), ms.float32))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Dense(self.num_patches * d_model, prediction_length)

    def _patchify(self, x: Tensor) -> Tensor:
        patches = []
        for idx in range(self.num_patches):
            start = idx * self.stride
            end = start + self.patch_length
            patches.append(x[:, start:end].expand_dims(1))
        return ops.concat(patches, axis=1)

    def construct(self, past_target: Tensor, past_time_feat: Tensor | None = None) -> Tensor:
        del past_time_feat
        context = past_target[:, -self.context_length :]
        scaled, loc, scale = self.scaler(context)
        patches = self._patchify(scaled)
        x = self.patch_proj(patches) + self.pos_embedding
        x = self.encoder(x)
        x = x.reshape(x.shape[0], -1)
        pred = self.head(x)
        return pred * scale + loc


class ForecastMSELoss(nn.Cell):
    def __init__(self, model: nn.Cell):
        super().__init__()
        self.model = model

    def construct(self, past_target: Tensor, past_time_feat: Tensor, future_target: Tensor) -> Tensor:
        pred = self.model(past_target, past_time_feat)
        return ops.mean(ops.square(pred - future_target))
