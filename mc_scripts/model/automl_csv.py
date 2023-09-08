"""
Don't forget to run ../../pipline/sample_run.sh
- Myeongchan
"""

# import os
# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/to/your/key/file'

from google.cloud import aiplatform
from google.cloud import storage
# from google.cloud import dataset

from benchmarks import dict_set_of_cols, config_generation_with_cmd_args, data_load, \
    data_split, filter_target_columns, get_dataset_name
from features import convert_gs_save_string

config = None
GS_ADDR_TEMPLATE = "gs://{bucket_name}/automl_dataset/df_automl_{dataset_name}.csv"


def main():
    # generate dataframe for automl: filter columns and fill null if needs
    df_automl = generate_df_for_automl(fill_null=config.fill_null)

    # upload df to google cloud bucket
    gc_path = upload_df_to_gcp_bucket(df_automl)
    return gc_path


def generate_df_for_automl(df=None, fill_null=False):
    global config

    if df is None:
        df_all = data_load(config)
    else:
        df_all = df.copy()

    df_automl = filter_target_columns(df_all, config.using_features, fill_null=fill_null)

    return df_automl


def get_gs_safe_dataset_name(config):
    dataset_name = get_dataset_name(config)
    dataset_name = convert_train_name_into_gs_path(dataset_name)
    return dataset_name


def get_gcs_address(config):
    global GS_ADDR_TEMPLATE
    bucket_name = config.bucket_name
    dataset_name = get_gs_safe_dataset_name(config)
    gc_path = GS_ADDR_TEMPLATE.format(bucket_name=bucket_name, dataset_name=dataset_name)
    return gc_path


def get_gs_safe_column_name(df):
    dict_col = dict()
    for col in df.columns:
        dict_col[col] = convert_gs_save_string(col)

    return df.rename(columns=dict_col)


def upload_df_to_gcp_bucket(df):
    global config
    dataset_name = get_gs_safe_dataset_name(config)

    df_safe = get_gs_safe_column_name(df)

    client = storage.Client()

    # Set up bucket name and file path
    bucket_name = config.bucket_name
    csv_file_path = f'automl_dataset/df_automl_{dataset_name}.csv'

    # Upload file to bucket
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(csv_file_path)
    blob.upload_from_string(df_safe.to_csv(index=False), 'text/csv')

    gc_path = get_gcs_address(config)
    print(f"Uploaded to {gc_path}")
    return gc_path


def convert_train_name_into_gs_path(dataset_name):
    dataset_name = dataset_name.replace("+", "plus")  # to satisfy gcp bucket naming rule
    return dataset_name


if __name__ == '__main__':
    config = config_generation_with_cmd_args()
    main()
