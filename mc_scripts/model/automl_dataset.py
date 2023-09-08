"""
Don't forget to run automl_csv.py before this script
- Myeongchan
"""

import os
from datetime import datetime

from parse import parse

from google.cloud import aiplatform

from automl_csv import get_gcs_address, GS_ADDR_TEMPLATE, config_generation_with_cmd_args

config = None
DISPLAY_NAME_FORMAT = "{bucket_name}_tableform_{dataset_name}"
PROJECT_NAME = 'shine-mobiledodctor'
REGION = 'asia-northeast3'


def main():
    global config
    gcs_path = get_gcs_address(config)
    try:
        ds = get_table_dataset(config)
        return ds
    except AssertionError as e:
        print("Not found dataset. Lets create! information:", e)
        print("Create dataset")

    ds = generate_df_vertex_dataset(gcs_path)
    print("Upload result: ", ds)
    return ds


def get_display_name(gcs_path):
    ret = parse(GS_ADDR_TEMPLATE, gcs_path)
    bucket_name = ret['bucket_name']
    dataset_name = ret['dataset_name']
    return DISPLAY_NAME_FORMAT.format(bucket_name=bucket_name, dataset_name=dataset_name, )


def generate_df_vertex_dataset(gcs_path):
    global config

    display_name = get_display_name(gcs_path)
    try:
        dataset = aiplatform.TabularDataset.create(
            display_name=display_name,
            gcs_source=[gcs_path],
            project=PROJECT_NAME,
            location=REGION,
        )
        return dataset
    except Exception as e:
        print("Error:", e)

        return None


def get_table_dataset(config):
    gcs_path = get_gcs_address(config)
    print("GCS path:", gcs_path)

    ret = parse(GS_ADDR_TEMPLATE, gcs_path)
    bucket_name = ret['bucket_name']
    dataset_name = ret['dataset_name']

    ds_display_name = DISPLAY_NAME_FORMAT.format(
        bucket_name=bucket_name,
        dataset_name=dataset_name)
    print(f"Dataset Display name: {ds_display_name}")

    # Load tabluear dataset
    filter = f'display_name="{ds_display_name}"'
    list_ds = aiplatform.TabularDataset.list(
        filter=filter,
        location=REGION,
    )
    print(f"Datasets:{list_ds}")
    assert list_ds, f"No dataset was found with setting region:{REGION}, filter={filter}"
    assert len(list_ds) == 1, f"One dataset must selected."
    return list_ds[0]


if __name__ == '__main__':
    config = config_generation_with_cmd_args()
    main()

