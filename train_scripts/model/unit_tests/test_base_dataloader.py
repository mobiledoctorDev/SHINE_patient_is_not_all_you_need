import os
import logging
import pytest

import numpy as np
import json

from mc_scripts.model.base_dataloader import IsraelDataBatchWithIntensity, IsraelDatasetWithIntensity, IsraelDataLoader


logger = logging.getLogger(__name__)


@pytest.fixture
def sample_train_gt():
    return np.load(f'{os.path.dirname(os.path.abspath(__file__))}/sample_train_gt.npy')


@pytest.fixture
def sample_train_input_cat():
    return np.load(f'{os.path.dirname(os.path.abspath(__file__))}/sample_train_input_cat.npy')


@pytest.fixture
def sample_train_input_ins():
    return np.load(f'{os.path.dirname(os.path.abspath(__file__))}/sample_train_input_ins.npy')


@pytest.fixture
def sample_dict_category():
    return json.load(
        open(f'{os.path.dirname(os.path.abspath(__file__))}/sample_dict_category.json', 'r')
    )


@pytest.fixture
def sample_israel_dataset(sample_train_gt, sample_train_input_cat, sample_train_input_ins, sample_dict_category):
    assert sample_train_gt.shape[0] == 17
    dataset = IsraelDatasetWithIntensity(
        sample_train_input_cat,
        sample_train_input_ins,
        sample_train_gt,
        sample_dict_category
    )
    return dataset


def test_1_dataset(sample_train_gt, sample_israel_dataset):
    dataset = sample_israel_dataset
    assert len(dataset) == sample_train_gt.shape[0]
    assert dataset.shape == [(17, 9), (17, 9), (17, 1)]


def test_1_1_dataset_getitem(sample_israel_dataset):
    dataset = sample_israel_dataset
    batch = dataset[[0, 1]]
    assert type(batch) == IsraelDataBatchWithIntensity
    assert batch.shape[0] == (2, 9)


def test_2_generator(sample_israel_dataset):

    data_loader = IsraelDataLoader(sample_israel_dataset, batch_size=16, shuffle=False)

    for batch_ndx, batch in enumerate(data_loader):
        logger.debug(f"batch_id:{batch_ndx}, batch_shape:{batch.shape}")

        if batch_ndx == 0:
            assert batch.shape == ((16, 9), (16, 9), (16, 1)), 'batch shape should be same as the sample numpy files'
        if batch_ndx == 1:
            assert batch.shape == ((1, 9), (1, 9), (1, 1)), 'second (last) batch length == 16'


def test_2_generator_shuffle(sample_israel_dataset):
    dataset = sample_israel_dataset

    np.random.seed(1212)
    data_loader = IsraelDataLoader(dataset, batch_size=4, shuffle=True)
    for batch_ndx, batch in enumerate(data_loader):
        logger.debug(f"batch_id:{batch_ndx}, batch:{batch}")

        if batch_ndx == 0:
            assert batch.shape == ((4, 9), (4, 9), (4, 1)), 'batch shape should be same as the sample numpy files'

        if batch_ndx == 4:
            assert batch.shape == ((1, 9), (1, 9), (1, 1)), 'batch shape should be same as the sample numpy files'
