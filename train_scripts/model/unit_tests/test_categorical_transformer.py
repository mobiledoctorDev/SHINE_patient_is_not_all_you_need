import logging

import torch
from torch import nn, optim
from sklearn.metrics import roc_auc_score, plot_roc_curve
from sklearn import svm

from mc_scripts.model.attention_model import CategoricalTransformer
from mc_scripts.model.utils import run_validation

from .test_base_dataloader import sample_train_input_ins, sample_train_gt, sample_train_input_cat, \
    sample_dict_category, sample_israel_dataset, IsraelDataLoader

logger = logging.getLogger(__name__)


def test_1_validation(sample_israel_dataset):
    train_gt = torch.FloatTensor(sample_israel_dataset.dataset_np_gt)
    train_cat = torch.LongTensor(sample_israel_dataset.dataset_np_cat)
    train_intensity = torch.FloatTensor(sample_israel_dataset.dataset_np_intensity)
    train_input = (train_cat, train_intensity)

    d_model = 64
    n_head = 2
    dict_category = sample_israel_dataset.dict_category
    loss_func = nn.BCELoss()

    transformer_model = nn.Transformer(
        d_model=d_model,
        nhead=n_head,
        num_encoder_layers=train_cat.shape[1]
    )
    model = CategoricalTransformer(
        transformer_model=transformer_model,
        dict_category=dict_category,

    )
    model.eval()
    val_metrics = {
        "loss": loss_func,
        'auc': lambda pred, gt: roc_auc_score(gt.numpy(), pred.numpy())
    }
    val_result = run_validation(model, val_metrics, train_input, train_gt)
    assert "loss" in val_result.keys()
    assert "auc" in val_result.keys()


def test_2_train_on_batch(sample_israel_dataset):
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

    transformer_model = nn.Transformer(
        d_model=d_model,
        nhead=n_head,
        num_encoder_layers=train_cat.shape[1]
    )
    model = CategoricalTransformer(
        transformer_model=transformer_model,
        dict_category=dict_category,

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


def test_2_train_on_epoch(sample_israel_dataset):
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

    transformer_model = nn.Transformer(
        d_model=d_model,
        nhead=n_head,
        num_encoder_layers=train_cat.shape[1]
    )
    model = CategoricalTransformer(
        transformer_model=transformer_model,
        dict_category=dict_category,

    )
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # **************** Train Epoch 0 ****************
    data_loader = IsraelDataLoader(sample_israel_dataset, batch_size=8, shuffle=True)
    train_result_info = CategoricalTransformer.train_on_epoch(0, data_loader, model, loss_func, optimizer)
    logger.debug(f"epoch 0 train finished. train_result_info:{train_result_info}")

    # ************** Check result after one batch *********************
    model.eval()
    val_result = run_validation(model, val_metrics, train_input, train_gt)
    first_epoch_loss = val_result['loss']
    logger.info(f"epoch 0 auc: {val_result['auc']}")

    # **************** Train Epoch 1 ****************
    data_loader = IsraelDataLoader(sample_israel_dataset, batch_size=8, shuffle=True)
    train_result_info = CategoricalTransformer.train_on_epoch(1, data_loader, model, loss_func, optimizer)
    logger.debug(f"epoch 1 train finished. train_result_info:{train_result_info}")

    # ************** Check result after one batch *********************
    model.eval()
    val_result = run_validation(model, val_metrics, train_input, train_gt)
    second_epoch_loss = val_result['loss']

    logger.info(f"epoch 1 auc: {val_result['auc']}")
    logger.info(f"first_epoch_loss:{first_epoch_loss}, second_epoch_loss:{second_epoch_loss}")
    assert first_epoch_loss != second_epoch_loss, 'Should learn something'


def test_3_val_on_epoch(sample_israel_dataset):
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

    transformer_model = nn.Transformer(
        d_model=d_model,
        nhead=n_head,
        num_encoder_layers=train_cat.shape[1]
    )
    model = CategoricalTransformer(
        transformer_model=transformer_model,
        dict_category=dict_category,

    )

    data_loader = IsraelDataLoader(sample_israel_dataset, batch_size=8, shuffle=True)
    result = CategoricalTransformer.validation_on_epoch(0, model, data_loader, val_metrics)
    logger.info(f"result:{result}")
