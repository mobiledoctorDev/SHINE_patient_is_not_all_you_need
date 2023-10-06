"""주어진 dataframe에 {baby_id}__{date_str} 을 이용해여 location 정보를 추가하는 함수를 작성.
"""
import os
import math
import joblib
from argparse import ArgumentParser

import pandas as pd
import numpy as np
from sklearn.preprocessing import QuantileTransformer
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

from utils import mkdir_if_not_exists
from calc_distance import distance_with_angle

DEFAULT_GPS_DATA_SOURCE = "./tmp_output/gps_full_2023-04-13.csv"
DEFAULT_BTS_DATA_SOURCE = "./tmp_output/bts_full_2023-04-13.csv"


def preprocess_run(df, prefix='gps'):
    generate_std_in_meter(df)

    target_columns = ['n', 'loc_std', 'path_sum']
    for col in target_columns:
        sr = df[col]
        sr.name = prefix + "_" + sr.name
        df[col + '_norm'] = normalize(sr, display=False)
        df[col + '_mask'] = get_mask_sr(df[col])

    # return results
    result_columns = ['index'] + target_columns + \
                     [col + '_norm' for col in target_columns] + \
                     [col + '_mask' for col in target_columns]
    return df[result_columns]


def generate_std_in_meter(df):
    # Define the coordinates of the two points in degrees
    lat1, lon1 = 37.0, -127.3
    lat2, lon2 = 37.0, -127.4

    distance_ratio_lon = distance_with_angle(lat1, lon1, lat2, lon2) * 10
    # Print the distance
    print(f"standardized 1 degree in longitude is {distance_ratio_lon:.2f} meters")

    # Define the coordinates of the two points in degrees
    lat1, lon1 = 37.1, -127.4
    lat2, lon2 = 37.0, -127.4

    distance_ratio_lat = distance_with_angle(lat1, lon1, lat2, lon2) * 10
    # Print the distance
    print(f"standardized 1 degree in latitude is {distance_ratio_lat:.2f} meters")

    df['lati_std_in_meter'] = distance_ratio_lat * df['lati_std']
    df['long_std_in_meter'] = distance_ratio_lon * df['long_std']
    df['loc_std'] = np.sqrt((df['lati_std_in_meter'] ** 2 + df['long_std_in_meter'] ** 2) / 2)
    return df


def norm_func_for_deployment(x, max_val, mean, std):
    if x == 0:
        return 0
    else:
        log_x = math.log(x + 1, 10)
        clip_x = min(log_x, max_val)
        return (clip_x - mean) / std


def normalize(sr, display=True, save_params=True, save_figure=True):
    # create long-tailed distribution data
    data = sr.values

    # normalize the data using Quantile Transformation
    qt = QuantileTransformer(output_distribution='normal')
    data_normalized = qt.fit_transform(data.reshape(-1, 1))

    # plot the original and normalized distributions
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].hist(data, bins=100)
    axs[0].set_title('Original Data Distribution')
    axs[1].hist(data_normalized, bins=100)
    axs[1].set_title('Normalized Data Distribution')
    if display:
        plt.show()
    if save_figure:
        mkdir_if_not_exists(f'./tmp_output/{bucket_name}/pp_params/')
        plt.savefig(f'./tmp_output/{bucket_name}/pp_params/{sr.name}_norm.png')
    plt.clf()

    if save_params:
        mkdir_if_not_exists(f'./tmp_output/{bucket_name}/pp_params/')
        # normalize the data using Quantile Transformation and save the transformer object
        data_normalized = qt.fit_transform(data.reshape(-1, 1))
        joblib.dump(qt, f'./tmp_output/{bucket_name}/pp_params/{sr.name}_qt.pkl')

    return data_normalized


def get_mask_sr(sr):
    return ((sr.isnull()) | (sr == 0)).astype(int)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--bucket_name", type=str, help="Please write bucket name")
    args = parser.parse_args()
    bucket_name = args.bucket_name

    # Preprocess GPS data
    print(f"load gps data from {DEFAULT_GPS_DATA_SOURCE} ...")
    df_gps = pd.read_csv(DEFAULT_GPS_DATA_SOURCE)
    df_gps_pp = preprocess_run(df_gps, prefix='gps')
    result_file_path = f"./tmp_output/{bucket_name}/gps_full_2023-04-13_norm.csv"
    df_gps_pp.to_csv(result_file_path, index=False)
    print(f"Save preprocessed GPS data to {result_file_path}.")

    # Preprocess BTS data
    print(f"load bts data from {DEFAULT_BTS_DATA_SOURCE} ...")
    df_bts = pd.read_csv(DEFAULT_BTS_DATA_SOURCE)
    df_bts_pp = preprocess_run(df_bts, prefix='bts')
    result_file_path = f"./tmp_output/{bucket_name}/bts_full_2023-04-13_norm.csv"
    df_bts_pp.to_csv(result_file_path, index=False)
    print(f"Save preprocessed BTS data to {result_file_path}.")
