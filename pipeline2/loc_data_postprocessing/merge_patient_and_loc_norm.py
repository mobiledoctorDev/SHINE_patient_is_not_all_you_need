import pandas as pd
from argparse import ArgumentParser

from utils import mkdir_if_not_exists

patient_id_col = 'patient_id'
selfcheck_date_col = 'selfcheck_date'


def add_loc_n_data(df, df_loc=None, n_days=0, prefix=''):
    # get a day before from self_check_date
    df[f'{selfcheck_date_col}_-{n_days}'] = pd.to_datetime(df[selfcheck_date_col]) - pd.Timedelta(days=n_days)
    df[f'{prefix}_loc_index-{n_days}'] = df[patient_id_col].astype(str) + '__' + df[f'{selfcheck_date_col}_-{n_days}'].astype(
        str)

    if df_loc is None:
        df_loc = pd.read_csv(BTS_DATA_RESULT)

    df_merged = pd.merge(
        df, df_loc,
        how='left', left_on=f'{prefix}_loc_index-{n_days}', right_on='index',
        suffixes=('', f'_{prefix}-{n_days}')
    )
    mask_cols = [x for x in df_merged.columns if x.endswith(f'_mask-{n_days}')]
    for mask_col in mask_cols:
        # fill 0 the value column
        norm_col = mask_col.replace(f'_mask-{n_days}', f'_norm-{n_days}')
        df_merged[norm_col] = df_merged[norm_col].fillna(0)

        # fill 1 for the mask columns
        df_merged[mask_col] = df_merged[mask_col].fillna(1)

    return df_merged


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--bucket_name", type=str, help="Please write bucket name")
    args = parser.parse_args()
    bucket_name = args.bucket_name

    df = pd.read_csv(f"./tmp_output/{bucket_name}/df_all_added_owid.csv")
    df_result = df.copy()

    GPS_DATA_RESULT = f"./tmp_output/{bucket_name}/gps_full_2023-04-13_norm.csv"
    print(f"load loc data from {GPS_DATA_RESULT} ...")
    df_pp = pd.read_csv(GPS_DATA_RESULT)
    for i in range(0, 8):
        df_result = add_loc_n_data(df_result, df_pp, i, prefix='gps')

    BTS_DATA_RESULT = f"./tmp_output/{bucket_name}/bts_full_2023-04-13_norm.csv"
    print(f"load loc data from {BTS_DATA_RESULT} ...")
    df_pp = pd.read_csv(BTS_DATA_RESULT)
    for i in range(0, 8):
        df_result = add_loc_n_data(df_result, df_pp, i, prefix='bts')

    mkdir_if_not_exists("./tmp_output")
    to_filepath = f"./tmp_output/{bucket_name}/df_all_added_owid_loc.csv"
    print(f"save result to {to_filepath} ...")
    df_result.to_csv(to_filepath, index=False)

    print("Final columns:", df_result.columns)
