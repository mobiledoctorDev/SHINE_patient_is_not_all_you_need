import os, sys
import math, copy, time
import logging

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, plot_roc_curve
from sklearn import svm
import torch
from torch import nn, optim
from torch.utils import data

from .utils import run_validation
from .base_model import McBaseAttentionModel

logger = logging.getLogger(__name__)


class CategoricalAttentionSimpleModel(McBaseAttentionModel):

    def __init__(self, dict_category, d_model=10, n_head=5, len_seq=9):
        """dict_category: 카테고리 의미 사전. GT로 쓸 값들은 포함하지 않는것이 좋음."""
        super().__init__()
        self.d_model = d_model
        self.len_seq = len_seq
        self.n_head = n_head

        # Categorical embedding
        self.dict_category = dict_category
        self.cat_embedding = nn.Linear(len(dict_category), d_model)

        # ffn for attention
        self.fc_key = nn.Linear(d_model, d_model)
        self.fc_query = nn.Linear(d_model, d_model)
        self.fc_value = nn.Linear(d_model, d_model)

        # multi-head attention
        self.multihead_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_head)

        # Final layer
        # self.final_fc = nn.Linear(len_seq * d_model, 1)
        self.final_fc = nn.Linear(d_model, 1)
        self.final_sigmoid = nn.Sigmoid()

    def forward(self, batch_tensor):
        """
        batch_tensor: (batch_size, seq_len)
        """

        # print("batch_tensor.shape:", batch_tensor.shape, batch_tensor[0, :])  # this should be [CLS] token
        one_hot_vector = nn.functional.one_hot(batch_tensor, num_classes=len(self.dict_category))
        # print("one_hot_vector.shape:", one_hot_vector.shape, one_hot_vector[0, 0, :])  # this should be [CLS] token
        one_hot_vector = one_hot_vector.float()
        cat_embed = self.cat_embedding(one_hot_vector)  # (batch_size, seq_len, dim)
        # print("cat_embed.shape:", cat_embed.shape)

        key = self.fc_key(cat_embed)
        query = self.fc_query(cat_embed)
        value = self.fc_value(cat_embed)

        attn_output, attn_output_weights = self.multihead_attn(query, key, value)
        # print("attn_output.shape:", attn_output.shape)

        # out = torch.flatten(attn_output, start_dim=1, end_dim=-1)
        # print("flattened. out.shape:", out.shape)
        # Get first vector of out tensor
        first_token_tensor = attn_output[:, 0, :]
        # print("first_token_tensor.shape:", first_token_tensor.shape)
        out = self.final_fc(first_token_tensor)
        out = self.final_sigmoid(out)
        # print("final tmp_output after sigmoid. out.shape:", out.shape)
        return out, attn_output_weights

    @classmethod
    def collate_fn(cls, batch):
        batch_cat = torch.LongTensor(batch.cat_values)
        batch_gt = torch.FloatTensor(batch.gt_data)
        # print("batch_gt:", batch_gt)
        return batch_cat, batch_gt
    #
    # @staticmethod
    # def train_on_epoch(cur_epoch, train_data_iterator, model, loss_func, optimizer):
    #     """각 모델에 맞게 배치를 적절히 변환해서 학습을 진행시키는 Trainer입니다.
    #     일단 급한대로 함수로 구현하고 추후에 Trainer 클래스로 바꾸고 공통된 부분은 부모나 공통함수로 빼면 되겠습니다. """
    #
    #     start = time.time()
    #     total_tokens = 0
    #     total_loss = 0
    #
    #     model.train()
    #     for i, batch in enumerate(train_data_iterator):
    #         train_batch_input = torch.LongTensor(batch.cat_values)
    #         train_batch_gt = torch.FloatTensor(batch.gt_data)
    #
    #         # ******************* train! *********************
    #         out, attn_vals = model(train_batch_input)
    #         #     print("Final_out:", out, "Gt:", train_batch_gt)
    #         loss_value = loss_func(out, train_batch_gt)
    #         # print("epoch:", cur_epoch, "batch:", i, "loss:", loss_value)
    #         total_loss += loss_value
    #         total_tokens += len(train_batch_input[0])
    #         loss_value.backward()
    #         optimizer.step()
    #         # **************************************************
    #
    #     total_train_loss_by_token = total_loss / total_tokens
    #     elapsed = time.time() - start
    #     logger.info(f"Epoch Step: {cur_epoch} Train Loss: {total_train_loss_by_token} elapsed: {elapsed}")
    #     return {
    #         "total_n_batch": i,
    #         "total_train_loss_by_token": total_train_loss_by_token,
    #         "total_tokens": total_tokens
    #     }
