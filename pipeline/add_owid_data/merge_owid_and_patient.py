import os

from argparse import ArgumentParser
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import pandas as pd
pd.set_option("display.max_column", 200)
pd.set_option("display.max_row", 200)

import seaborn
seaborn.set()

from utils import mkdir_if_not_exists


parser = ArgumentParser()
parser.add_argument("--bucket_name", type=str, help="Please write bucket name")
args = parser.parse_args()
bucket_name = args.bucket_name
assert bucket_name is not None

shine_filepath = f'../output/{bucket_name}/df_all.csv'
print(f"Loading shine data from {shine_filepath} ...")
df_shine = pd.read_csv(shine_filepath)
print(df_shine.shape, df_shine.columns)

source_check_col = 'selfcheck_date'
selfcheck_col = 'selfcheck_date'
df_shine = df_shine.rename(columns={source_check_col: selfcheck_col})
print(df_shine[selfcheck_col].min(), df_shine[selfcheck_col].max())

print(f"Loading owid data from ../resource/{bucket_name}/owid-covid-data.csv ...")
df_owid = pd.read_csv(f'../resource/{bucket_name}/owid-covid-data.csv')
print(df_owid.shape, df_owid.columns)

date_col_on_owid = 'date'
print(df_owid[date_col_on_owid].min(), df_owid[date_col_on_owid].max())

# Filter same date of shine data
df_owid = df_owid[df_owid[date_col_on_owid] < df_shine[selfcheck_col].max()]
print(f"After filtering, owid data has {df_owid.shape[0]} rows.")

# Preprocess owid data
# Add special columns
from datetime import timedelta
df_owid['date_index'] = df_owid['date'] + "__" + df_owid['iso_code']
df_owid['date_index_6months_ago'] = (pd.to_datetime(df_owid['date']) - timedelta(days=182)).dt.strftime("%Y-%m-%d")
df_owid['date_index_6months_ago'] = df_owid['date_index_6months_ago'] + "__" + df_owid['iso_code']


def get_values_for_6months(df, target_col):
    return df[target_col] - pd.merge(
        df[['date_index', 'date_index_6months_ago', target_col]],
        df[['date_index', target_col]],
        how='left',
        left_on='date_index_6months_ago',
        right_on='date_index',
        suffixes=("", '_6months_ago'),
    )[f'{target_col}_6months_ago']


# 6개월 확진자 수
df_owid['total_cases_per_million_for_6months'] = get_values_for_6months(df_owid, 'total_cases_per_million' )

# 6개월 백신접종자 수
df_owid['total_vaccinations_per_hundred_for_6months'] = get_values_for_6months(df_owid, 'total_vaccinations_per_hundred')

# 6개월 사망자 수
df_owid['total_deaths_per_million_for_6months'] = get_values_for_6months(df_owid, 'total_deaths_per_million')


# Select columns
per_pop_cols = [x for x in df_owid.columns if "per_million" in x or 'per_hundred' in x or 'per_thousand' in x]
hw_cols = ['positive_rate', 'reproduction_rate']
additional_cols = per_pop_cols + hw_cols
print(f"Selected columns: {additional_cols}")
df_owid_target = df_owid[['iso_code', date_col_on_owid] + additional_cols]

df_owid_nm = df_owid_target[['iso_code', date_col_on_owid]].copy()
dict_max_vals = dict()
for col in additional_cols:
    norm_col = col + '_norm'
    max_val = df_owid_target[col].max()
    df_owid_nm[norm_col] = df_owid_target[col] / max_val
    print(f"max value of {col}: {max_val}")
    dict_max_vals[col] = max_val

    col_mask = col + '_mask'
    df_owid_nm[col_mask] = df_owid_nm[norm_col].isnull().astype(int)
    df_owid_nm[norm_col] = df_owid_nm[norm_col].fillna(0)

# Select Korea's
df_kr = df_owid_nm[df_owid_nm['iso_code'] == 'KOR']
print(df_kr.shape, df_kr[date_col_on_owid].min(), df_kr[date_col_on_owid].max())
df_kr.sample(5).sort_values(date_col_on_owid)

# Merge shine data and owid data
df_new = pd.merge(df_shine, df_kr, how='left', left_on=selfcheck_col, right_on='date')
print(f"The final result has {df_new.shape[0]} rows.")

# Save the result
save_filepath = f'../output/{bucket_name}/df_all_added_owid.csv'
df_new.to_csv(save_filepath, index=False)
print(f"Saved to {save_filepath}.")

# Confirm the result
df_load = pd.read_csv(save_filepath)
print(df_load.sample(5))
print("Null check:", df_load.isnull().sum(axis=0))
print("Final columns: ", df_load.columns)
