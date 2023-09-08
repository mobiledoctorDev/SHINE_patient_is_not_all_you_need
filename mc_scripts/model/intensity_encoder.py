import math
from typing import Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class IntensityEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_value: int = 1):
        super().__init__()

        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        self.max_value = max_value
        self.max_len = max_len = 5000
        self.multiplier = max_len / max_value

    def forward(self, x: Tensor, intensity_tensor: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
            intensity_tensor: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        div_term = torch.exp(torch.arange(0, self.d_model, 2) * (-math.log(10000.0) / self.d_model))
        pe = torch.zeros(self.max_len, 1, self.d_model)
        pe[:, 0, 0::2] = torch.sin(intensity_tensor * div_term)
        pe[:, 0, 1::2] = torch.cos(intensity_tensor * div_term)

        x = x + pe[:x.size(0)]
        return self.dropout(x)


class IntensitySimpleMultiplier(nn.Module):
    def forward(self, x: Tensor, intensity_tensor: Tensor) -> Tensor:
        return x * intensity_tensor


class ScaledIntensityAdder(nn.Module):
    def __init__(self, d_model:int, dropout: float = 0.1, max_divisor_scale: int = 10):
        super().__init__()

        self.d_model = d_model
        self.max_divisor = max_divisor_scale
        self.dropout = nn.Dropout(p=dropout)

        # Set intensity scaler
        step = (max_divisor_scale - 1) / (d_model - 1)
        series_i = torch.arange(1, max_divisor_scale + step, step)
        series_i = series_i[series_i <= max_divisor_scale]
        self.scaler = 1 / torch.pow(2, series_i)
        self.scaler[0::2] = self.scaler[0::2]
        self.scaler[1::2] = self.scaler[1::2] * -1

    def forward(self, x: Tensor, intensity_tensor: Tensor) -> Tensor:
        scaled_intensity = intensity_tensor * self.scaler
        x = x + scaled_intensity
        return self.dropout(x)
