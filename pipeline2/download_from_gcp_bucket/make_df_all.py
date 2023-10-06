import os, sys
from argparse import ArgumentParser
import pandas as pd

from utils import mkdir_if_not_exists

def load_data(dataset_name):
    """ dataset name could be "train", "valid" , 'test', """
    df_filepath = [x for x in os.listdir(source_dir) if x.startswith("df_") and dataset_name in x]
    assert len(df_filepath) == 1
    df = pd.read_csv(os.path.join(source_dir, df_filepath[0]))
    print(dataset_name, df.shape)
    return df


parser = ArgumentParser()
parser.add_argument("--bucket_name", type=str, help="Please write bucket name")
parser = parser.parse_args()

source_dir = "../resource/" + parser.bucket_name
result_dir = '../output/' + parser.bucket_name

df_train = load_data("train")
df_val = load_data("valid")
df_test = load_data("test")

n_total = df_train.shape[0] + df_val.shape[0] + df_test.shape[0]
print(f"n_total: {n_total}")
print(f"ratio: {df_train.shape[0] / n_total:.1%}, {df_val.shape[0] / n_total:.1%}, {df_test.shape[0] / n_total:.1%}")

df_train['split'] = 'TRAIN'
df_val['split'] = 'VALIDATE'
df_test['split'] = 'TEST'

df_all = pd.concat([df_train, df_val, df_test], axis=0)
print("df_all.shape: ", df_all.shape)

print("df_all.columns: ", df_all.columns)
print("sample: ")
print(df_all.sample(5))

mkdir_if_not_exists(result_dir)
print(f"Saving to {os.path.join(result_dir, 'df_all.csv')} ...")
df_all.to_csv(os.path.join(result_dir, "df_all.csv"), index=False)
