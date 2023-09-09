"""
Don't forget to run automl_dataset.py before this script
- Myeongchan
"""
import os
from datetime import datetime

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/mckim/Key/shine-mobiledodctor-xxx.json"

from google.cloud import aiplatform
from google.cloud import storage
from parse import parse
import pandas as pd
from parse import parse

from benchmarks import config_generation_with_cmd_args
from automl_dataset import DISPLAY_NAME_FORMAT, GS_ADDR_TEMPLATE, get_gcs_address, get_table_dataset
from features import column_specs
from automl_train_result import save_fig

config = None

PROJECT_NAME = 'shine-mobiledodctor'
REGION = 'asia-northeast3'


def main():
    print(f"Let's try to get dataset. config: {config}")
    ds = get_table_dataset(config)

    assert config.training_num is not None, "Must set training number i"
    i = config.training_num
    date_string = datetime.now().strftime('%Y%m%d')
    if i is None:
        model_display_name = ds.display_name + f"_{date_string}"
    else:
        model_display_name = ds.display_name + f"_{date_string}_{i}"

    print_ds_info(ds)
    target_col_spec = {k: v for k, v in column_specs.items() if k in ds.column_names}
    print(f"target_col_spec: {target_col_spec}")

    model = run_automl_training(
        dataset=ds,
        col_specs=target_col_spec,
        display_name=model_display_name
    )

    # res = model.get_model_evaluation()
    # print(res)
    # save_fig(res)


def run_automl_training(dataset, col_specs, display_name):
    job = aiplatform.AutoMLTabularTrainingJob(
        display_name=display_name,
        optimization_prediction_type='classification',
        optimization_objective='maximize-au-roc',
        column_specs=col_specs,
        #     column_transformations: Optional[List[Dict[str, Dict[str, str]]]] = None,
        #     optimization_objective_recall_value: Optional[float] = None,
        #     optimization_objective_precision_value: Optional[float] = None,
        project=PROJECT_NAME,
        location=REGION,
        #     credentials: Optional[google.auth.credentials.Credentials] = None,
        #     labels: Optional[Dict[str, str]] = None,
        #     training_encryption_spec_key_name: Optional[str] = None,
        #     model_encryption_spec_key_name: Optional[str] = None,
    )

    model = job.run(
        dataset=dataset,
        target_column='pcr_result',
        #     training_fraction_split: Optional[float] = None,
        #     validation_fraction_split: Optional[float] = None,
        #     test_fraction_split: Optional[float] = None,
        predefined_split_column_name='split',
        #     timestamp_split_column_name: Optional[str] = None,
        #     weight_column: Optional[str] = None,
        budget_milli_node_hours=2000,
        #     model_display_name: Optional[str] = None,
        #     model_labels: Optional[Dict[str, str]] = None,
        #     model_id: Optional[str] = None,
        #     parent_model: Optional[str] = None,
        #     is_default_version: Optional[bool] = True,
        #     model_version_aliases: Optional[Sequence[str]] = None,
        #     model_version_description: Optional[str] = None,
        disable_early_stopping=False,
        #     export_evaluated_data_items: bool = False,
        #     export_evaluated_data_items_bigquery_destination_uri: Optional[str] = None,
        #     export_evaluated_data_items_override_destination: bool = False,
        #     additional_experiments: Optional[List[str]] = None,
        sync=False,
        #     create_request_timeout: Optional[float] = None,
    )
    return model


def print_ds_info(ds):
    from benchmarks import SYSTEM_COLS, GT_COL
    except_cols = SYSTEM_COLS + [GT_COL]
    print(f"Dataset.display_name:{ds.display_name}")
    print(f"Dataset.column_name:{ds.column_names}")
    for col in ds.column_names:
        if col in except_cols:
            continue
        assert col in column_specs, f"No definition of column:{col}. Please define it in column_spec of this source."
    print("Column specs:", len(ds.column_names))


if __name__ == '__main__':
    config = config_generation_with_cmd_args()
    main()
