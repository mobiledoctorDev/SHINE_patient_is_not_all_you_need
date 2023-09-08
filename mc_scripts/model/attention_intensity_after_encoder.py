import math, copy, time
import logging

import torch
from torch import nn, optim

from .base_model import McBaseAttentionModel
from .intensity_encoder import ScaledIntensityAdder
from .attention_model import CategoricalAttentionLayer

logger = logging.getLogger(__name__)


class CategoricalAttentionEncoderModel(McBaseAttentionModel):
    def __init__(self, dict_category, d_model=10, n_head=5, len_seq=9, **kwargs_for_attention_layer):
        super().__init__()
        self.d_model = d_model
        self.len_seq = len_seq
        self.n_head = n_head

        # Categorical embedding
        self.dict_category = dict_category

        # Categorical embedding
        self.cat_embedding = nn.Embedding(len(dict_category), d_model)

        self.attention_layer = CategoricalAttentionLayer(d_model, n_head, **kwargs_for_attention_layer)

        # Final layer
        self.final_fc = nn.Linear(len_seq * d_model, 1)
        self.final_sigmoid = nn.Sigmoid()

    def forward(self, train_batch):
        """
        train_batch: [train_batch_category_value (batch_size, seq_len), train_batch_intensity (batch_size, seq_len)]
        """

        train_batch_category, train_batch_intensity = train_batch

        # print("self.cat_embedding:", self.cat_embedding)
        # print("train_batch:", train_batch_category)
        # print("train_batch_intensity:", train_batch_intensity.shape)
        # print("train_batch_intensity:", train_batch_intensity)
        cat_embed = self.cat_embedding(train_batch_category)
        # print('cat_embed.shape:', cat_embed.shape)  # (batch_size, seq_len, d_model)
        attn_out, attn_weights = self.attention_layer(cat_embed)

        intensity = train_batch_intensity.view(*train_batch_intensity.shape, 1)  # (batch_size, seq_len, 1)
        intensity = intensity.expand(*intensity.shape[:-1], self.d_model)  # (batch_size, seq_len, d_model)
        # print("intensity shape:", intensity.shape)

        out_with_intensity = self.intensity_apply(attn_out, intensity)

        out = torch.flatten(out_with_intensity, start_dim=1, end_dim=-1)
        # print("after flatten: out.shape:", out.shape)
        out = self.final_fc(out)
        out = self.final_sigmoid(out)

        return out, attn_weights

    def intensity_apply(self, input_tensor, intensity_tensor):
        raise NotImplementedError("You should define how to handle intensity values")

    @classmethod
    def collate_fn(cls, batch):
        train_batch_cat = torch.LongTensor(batch.cat_values)
        train_batch_intensity = torch.FloatTensor(batch.intensity_values)
        train_batch_input = (train_batch_cat, train_batch_intensity)
        train_batch_gt = torch.FloatTensor(batch.gt_data)

        return train_batch_input, train_batch_gt


class MultiplyIntensityModel(CategoricalAttentionEncoderModel):
    def intensity_apply(self, input_tensor, intensity_tensor):
        intensity_embeds = input_tensor * intensity_tensor
        return intensity_embeds
