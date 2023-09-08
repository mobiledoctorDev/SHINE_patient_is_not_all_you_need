import math, copy, time
import logging

import torch
from torch import nn, optim

from .base_model import McBaseAttentionModel
from .intensity_encoder import ScaledIntensityAdder

logger = logging.getLogger(__name__)


class CategoricalTransformerEncoder(McBaseAttentionModel):
    def __init__(self, transformer_encoder, dict_category, len_seq=9):
        super().__init__()
        assert len(transformer_encoder.layers) > 0, "At least one encoder layer exist"
        self.d_model = transformer_encoder.layers[0].self_attn.embed_dim
        self.len_seq = len_seq
        self.n_head = transformer_encoder.layers[0].self_attn.num_heads

        # Categorical embedding
        self.dict_category = dict_category

        # Category embeding layer
        self.cat_embeding = nn.Linear(len(dict_category), self.d_model)
        self.transformer_encoder = transformer_encoder

        # Final layer
        self.global_pooling = nn.AvgPool1d(len_seq)
        self.final_fc = nn.Linear(len_seq * self.d_model, 1)
        self.final_activation = nn.Sigmoid()

    def forward(self, train_batch):
        """
        train_batch: (batch_size, seq_len)
        train_batch_intensity: (batch_size, seq_len)
        """
        train_batch_category, train_batch_intensity = train_batch

        cat_embeds = self.cat_embeding(train_batch_category)
        intensity = train_batch_intensity.view(*train_batch_intensity.shape, 1)  # (batch_size, seq_len, 1)
        intensity = intensity.expand(*intensity.shape[:-1], self.d_model)  # (batch_size, seq_len, d_model)
        intensity_embeds = self.intensity_apply(cat_embeds, intensity)

        out = self.transformer_encoder(intensity_embeds)   # (batch size, seq_len, d_model)
        out = self.final_top(out)
        return out, None

    @classmethod
    def collate_fn(cls, batch):
        train_batch_cat = torch.LongTensor(batch.cat_values)
        train_batch_intensity = torch.FloatTensor(batch.intensity_values)
        train_batch_input = (train_batch_cat, train_batch_intensity)
        train_batch_gt = torch.FloatTensor(batch.gt_data)

        return train_batch_input, train_batch_gt

    def intensity_apply(self, input_tensor, intensity_tensor):
        raise NotImplementedError("You should define how to handle intensity values")

    def final_top(self, out):
        out = torch.flatten(out, start_dim=1, end_dim=-1)
        out = self.final_fc(out)
        out = self.final_activation(out)
        return out


class MuliplyIntensityUsingCls(CategoricalTransformerEncoder):
    def __init__(self, transformer_encoder, dict_category, len_seq=9):
        super().__init__(transformer_encoder, dict_category, len_seq)
        # Override the final fc layer
        self.final_fc = nn.Linear(self.d_model, 1)

    def intensity_apply(self, input_tensor, intensity_tensor):
        intensity_embeds = input_tensor * intensity_tensor
        return intensity_embeds

    def final_top(self, out):
        # Get first vector of out tensor
        first_token_tensor = out[:, 0, :]

        out = self.final_fc(first_token_tensor)
        out = self.final_activation(out)
        return out


class MultiplyIntensity(CategoricalTransformerEncoder):
    def intensity_apply(self, input_tensor, intensity_tensor):
        intensity_embeds = input_tensor * intensity_tensor
        return intensity_embeds


class AddIntensity(CategoricalTransformerEncoder):
    def intensity_apply(self, input_tensor, intensity_tensor):
        intensity_embeds = input_tensor + intensity_tensor
        return intensity_embeds


class AddScaledIntensityModel(CategoricalTransformerEncoder):
    def __init__(self, transformer_encoder, dict_category, len_seq=9, dropout=0.1):
        """Intensity is scaled by max_divisor_scale. Then, it is added to the input tensor."""
        super().__init__(transformer_encoder, dict_category, len_seq)
        self.intensity_encoder = ScaledIntensityAdder(
            d_model=self.d_model,
            max_divisor_scale=10,
            dropout=dropout
        )

    def intensity_apply(self, input_tensor, intensity_tensor):
        """Intensity it is added to the input tensor."""
        intensity_embeds = self.intensity_encoder(input_tensor, intensity_tensor)
        return intensity_embeds


class CategoricalTransformerEncoderIntensityAfterEncoding(McBaseAttentionModel):
    def __init__(self, transformer_encoder, dict_category, len_seq=9):
        super().__init__()
        assert len(transformer_encoder.layers) > 0, "At least one encoder layer exist"
        self.d_model = transformer_encoder.layers[0].self_attn.embed_dim
        self.len_seq = len_seq
        self.n_head = transformer_encoder.layers[0].self_attn.num_heads

        # Categorical embedding
        self.dict_category = dict_category

        # Category embeding layer
        self.cat_embeding = nn.Embedding(len(dict_category), self.d_model)
        self.transformer_encoder = transformer_encoder

        # Final layer
        self.final_fc = nn.Linear(len_seq * self.d_model, 1)
        self.final_sigmoid = nn.Sigmoid()

    def forward(self, train_batch):
        """
        train_batch: (batch_size, seq_len)
        train_batch_intensity: (batch_size, seq_len)
        """
        train_batch_category, train_batch_intensity = train_batch

        cat_embeds = self.cat_embeding(train_batch_category)
        out = self.transformer_encoder(cat_embeds)   # (batch size, seq_len, d_model)

        intensity = train_batch_intensity.view(*train_batch_intensity.shape, 1)  # (batch_size, seq_len, 1)
        intensity = intensity.expand(*intensity.shape[:-1], self.d_model)  # (batch_size, seq_len, d_model)
        out = self.intensity_apply(out, intensity)
        out = torch.flatten(out, start_dim=1, end_dim=-1)
        # print("after flatten: out.shape:", out.shape)
        out = self.final_fc(out)
        out = self.final_sigmoid(out)
        return out, None

    @classmethod
    def collate_fn(cls, batch):
        train_batch_cat = torch.LongTensor(batch.cat_values)
        train_batch_intensity = torch.FloatTensor(batch.intensity_values)
        train_batch_input = (train_batch_cat, train_batch_intensity)
        train_batch_gt = torch.FloatTensor(batch.gt_data)

        return train_batch_input, train_batch_gt

    def intensity_apply(self, input_tensor, intensity_tensor):
        raise NotImplementedError("You should define how to handle intensity values")


class MultiplyIntensityAfterEncoding(CategoricalTransformerEncoderIntensityAfterEncoding):
    def intensity_apply(self, input_tensor, intensity_tensor):
        intensity_embeds = input_tensor * intensity_tensor
        return intensity_embeds


class CategoricalTransformerEncoderAddIntensityAfterEncoding(CategoricalTransformerEncoderIntensityAfterEncoding):
    def intensity_apply(self, input_tensor, intensity_tensor):
        intensity_embeds = input_tensor + intensity_tensor
        return intensity_embeds


class CategoricalTransformerEncoderAddScaledIntensityAfterEncoding(CategoricalTransformerEncoderIntensityAfterEncoding):
    def __init__(self, transformer_encoder, dict_category, len_seq=9, dropout=0.1):
        """Intensity is scaled by max_divisor_scale. Then, it is added to the input tensor."""
        super().__init__(transformer_encoder, dict_category, len_seq)
        self.intensity_encoder = ScaledIntensityAdder(
            d_model=self.d_model,
            max_divisor_scale=10,
            dropout=dropout
        )

    def intensity_apply(self, input_tensor, intensity_tensor):
        """Intensity it is added to the input tensor."""
        intensity_embeds = self.intensity_encoder(input_tensor, intensity_tensor)
        return intensity_embeds