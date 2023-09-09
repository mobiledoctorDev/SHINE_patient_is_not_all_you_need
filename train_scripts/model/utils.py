import logging
from collections import defaultdict
from datetime import datetime, timedelta
import math, copy, time
import random

import pandas as pd


logger = logging.getLogger(__name__)


# Util functions
def size_and_ratio(df, col, dropna=True):
    if pd.__version__ > '1.1':
        sr = df.groupby(col, dropna=dropna).size().sort_values(ascending=False)
    elif dropna:
        sr = df.groupby(col).size().sort_values(ascending=False)
    else:  # dropna == False
        sr = df.fillna('nan').groupby(col).size().sort_values(ascending=False)
    sr_ratio = sr.copy() / sum(sr)
    print("Sum :", sum(sr), sr.shape)
    sr.name = 'size'
    sr_ratio.name = 'ratio'
    return pd.concat([sr, sr_ratio], axis=1)


def split_train_test_with_baby_id(df, target_col='baby_id', test_ratio=0.2, seed=1212):
    babies = df[target_col].unique().tolist()
    print("babies:", babies[:5])
    random.seed(seed)
    random.shuffle(babies)
    print("shuffled:", babies[:5])
    n_total = len(babies)
    n_test = int(n_total * test_ratio)
    n_train = n_total - n_test
    train_babies = babies[:n_train]
    test_babies = babies[n_train:]

    df_train_tmp = df[df[target_col].isin(train_babies)]
    df_test_tmp = df[df[target_col].isin(test_babies)]
    return df_train_tmp, df_test_tmp


def get_category_key(colname, type_name, value):
    if type_name == 'category':
        return f"{colname}_{str(float(value))}"
    elif type_name == 'float':
        return colname
    else:
        raise NotImplementedError(f"unkown type: {type_name}")


def guess_type(sr):
    if sr.nunique() > 10:
        type_name = 'float'
    else:
        type_name = 'category'
    return type_name


def get_dict_category_from_dataset(df_input):
    """generate dictionary about category information and type-specified dataframe. from a train dataset"""

    class CatValIndex:
        cat_val_idx = 0

        @classmethod
        def get_new_cat_val_idx(cls):
            """index create function. Add 1 to index when every index created"""
            cur_val = cls.cat_val_idx
            cls.cat_val_idx += 1
            return cur_val

    df_input_typed = df_input.copy()

    dict_category_tmp = defaultdict(CatValIndex.get_new_cat_val_idx)

    for col in df_input.columns:

        type_name = guess_type(df_input[col])
        print(col, type_name, df_input[col].unique()[:10]) \
            if df_input[col].nunique() > 10 \
            else print(col, type_name, df_input[col].unique()[:10], "...")

        df_input_typed[col] = df_input_typed[col].astype(type_name)
        if type_name == 'category':
            for val in df_input_typed[col].unique():
                dict_key = get_category_key(col, type_name, val)
                category_idx = dict_category_tmp[dict_key]
        elif type_name == 'float':
            dict_key = get_category_key(col, type_name, None)
            category_idx = dict_category_tmp[dict_key]
        else:
            raise NotImplementedError(f"unknown type: {type_name}")

    dict_category = dict(dict_category_tmp)
    return dict_category, df_input_typed


def run_validation(model, metrics, val_input, val_gt):
    val_pred, attn_val = model(val_input)

    result = dict()
    for key, metric_func in metrics.items():
        result[key] = metric_func(val_pred.to('cpu').detach(), val_gt.to('cpu').detach())
    return result


def create_cat_and_intensity_from_df(df_typed, dict_category):
    df_category = pd.DataFrame()
    df_intensity = df_category.copy()

    for index, row in df_typed.iterrows():
        for col, val in row.items():
            type_name = df_typed[col].dtype
            key = get_category_key(colname=col, type_name=type_name, value=val)
            cat_idx = dict_category[key]

            df_category.loc[index, col] = cat_idx
            if type_name == 'category':
                df_intensity.loc[index, col] = 1
            if type_name == 'float':
                df_intensity.loc[index, col] = val

    return df_category, df_intensity