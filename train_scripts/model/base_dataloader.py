import numpy as np

import torch
from torch import nn, optim
from torch.utils import data
from torch.utils.data.dataset import T_co


class IsraelDataBatchWithIntensity:
    def __init__(self, batch_cat_value, batch_intensity, batch_gt):
        self.cat_values = batch_cat_value
        self.intensity_values = batch_intensity
        self.gt_data = batch_gt

    @property
    def input_data(self):
        return (self.cat_values, self.intensity_values)

    @property
    def shape(self):
        return (self.cat_values.shape, self.intensity_values.shape, self.gt_data.shape)

    def __len__(self):
        return self.cat_values.shape[0]


class IsraelDatasetWithIntensity(data.Dataset):
    """Iterable Dataset"""

    def __init__(self, dataset_np_cat, dataset_np_intensity, dataset_np_gt, dict_category):
        super(IsraelDatasetWithIntensity).__init__()
        self.dataset_np_cat = dataset_np_cat
        self.dataset_np_intensity = dataset_np_intensity
        self.dataset_np_gt = dataset_np_gt
        self.dict_category = dict_category

        assert dataset_np_gt.shape[0] == dataset_np_cat.shape[0] == dataset_np_intensity.shape[0], \
            "cat_val, intensity, gt must have same length"

    @classmethod
    def from_csv(cls, csv_path):
        return cls(None, None, None, {})

    @classmethod
    def from_pickle(cls, pickle_path):
        return cls(None, None, None, {})

    def __getitem__(self, index) -> T_co:
        return IsraelDataBatchWithIntensity(
            self.dataset_np_cat[index, ...],
            self.dataset_np_intensity[index, ...],
            self.dataset_np_gt[index, ...]
        )

    def __len__(self):
        return self.dataset_np_cat.shape[0]

    @property
    def shape(self):
        return [self.dataset_np_cat.shape, self.dataset_np_intensity.shape, self.dataset_np_gt.shape]


class IsraelDataLoader(data.DataLoader):
    def __init__(self, dataset, batch_size=64, shuffle=False, **kwargs):
        super(IsraelDataLoader, self).__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=IsraelDataLoader.collate_fn,
            **kwargs)

    @staticmethod
    def collate_fn(batch):
        """Batch 클래스 인스턴스 하나를 numpy array 의 형태로 가지고 있게 반환합니다. """
        batch_cat_value = np.stack([b.cat_values for b in batch], axis=0)
        batch_intensity_value = np.stack([b.intensity_values for b in batch], axis=0)
        batch_gt_value = np.stack([b.gt_data for b in batch], axis=0)

        return IsraelDataBatchWithIntensity(batch_cat_value, batch_intensity_value, batch_gt_value)
