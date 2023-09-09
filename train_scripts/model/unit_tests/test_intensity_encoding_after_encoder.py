import logging
import math
import numpy

import torch
from torch import nn, optim
from sklearn.metrics import roc_auc_score, plot_roc_curve
from sklearn import svm

from mc_scripts.model.attention_intensity_after_encoder import MultiplyIntensityModel as IntensityAfterAttentionModel

from mc_scripts.model.utils import run_validation
from mc_scripts.model.intensity_encoder import ScaledIntensityAdder

from .test_base_dataloader import sample_train_input_ins, sample_train_gt, sample_train_input_cat, \
    sample_dict_category, sample_israel_dataset, IsraelDataLoader


logger = logging.getLogger(__name__)


def test_4_multiply_model_validation(sample_israel_dataset):
    train_gt = torch.FloatTensor(sample_israel_dataset.dataset_np_gt)
    train_cat = torch.LongTensor(sample_israel_dataset.dataset_np_cat)
    train_intensity = torch.FloatTensor(sample_israel_dataset.dataset_np_intensity)
    train_input = (train_cat, train_intensity)

    d_model = 64
    n_head = 2
    dict_category = sample_israel_dataset.dict_category
    loss_func = nn.BCELoss()

    model = IntensityAfterAttentionModel(
        dict_category, d_model=d_model, n_head=n_head, len_seq=train_cat.shape[1], dropout=0.1
    )
    model.eval()
    val_metrics = {
        "loss": loss_func,
        'auc': lambda pred, gt: roc_auc_score(gt.numpy(), pred.numpy())
    }
    val_result = run_validation(model, val_metrics, train_input, train_gt)
    assert "loss" in val_result.keys()
    assert "auc" in val_result.keys()


def test_5_multiply_model_train_on_batch(sample_israel_dataset):
    train_gt = torch.FloatTensor(sample_israel_dataset.dataset_np_gt)
    train_cat = torch.LongTensor(sample_israel_dataset.dataset_np_cat)
    train_intensity = torch.FloatTensor(sample_israel_dataset.dataset_np_intensity)
    train_input = (train_cat, train_intensity)

    # ********* Hyperparameters **************
    d_model = 64
    n_head = 2
    dict_category = sample_israel_dataset.dict_category
    loss_func = nn.BCELoss()
    val_metrics = {
        "loss": loss_func,
        'auc': lambda pred, gt: roc_auc_score(gt.numpy(), pred.numpy())
    }

    # Model and optimizer
    model = IntensityAfterAttentionModel(
        dict_category, d_model=d_model, n_head=n_head, len_seq=train_cat.shape[1], dropout=0.1
    )
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # ************** Check initial result *********************
    model.eval()
    val_result = run_validation(model, val_metrics, train_input, train_gt)
    init_loss = val_result['loss']

    # **************** Train ****************
    model.train()
    out, attn_output_weights = model(train_input)
    loss_value = loss_func(out, train_gt)
    loss_value.backward()
    optimizer.step()

    # ************** Check result after one batch *********************
    model.eval()
    val_result = run_validation(model, val_metrics, train_input, train_gt)
    trained_loss = val_result['loss']

    logger.debug(f"loss before:{init_loss}, loss after:{trained_loss}")
    assert init_loss != trained_loss, 'Should learn something'


def test_6_multiply_model_val_on_epoch(sample_israel_dataset):
    train_gt = torch.FloatTensor(sample_israel_dataset.dataset_np_gt)
    train_cat = torch.LongTensor(sample_israel_dataset.dataset_np_cat)
    train_intensity = torch.FloatTensor(sample_israel_dataset.dataset_np_intensity)
    train_input = (train_cat, train_intensity)

    # ********* Hyperparameters **************
    d_model = 64
    n_head = 2
    dict_category = sample_israel_dataset.dict_category
    loss_func = nn.BCELoss()
    val_metrics = {
        "loss": loss_func,
        'auc': lambda pred, gt: roc_auc_score(gt.numpy(), pred.numpy())
    }

    # ************** Check initial result *********************
    model = IntensityAfterAttentionModel(
        dict_category, d_model=d_model, n_head=n_head, len_seq=sample_israel_dataset.dataset_np_cat.shape[1], dropout=0.1
    )
    data_loader = IsraelDataLoader(sample_israel_dataset, batch_size=8, shuffle=True)

    result = IntensityAfterAttentionModel.validation_on_epoch(0, model, data_loader, val_metrics)
    logger.info(f"result:{result}")