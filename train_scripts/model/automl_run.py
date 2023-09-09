import automl_csv
import automl_dataset
import automl_train

import os
import time

from benchmarks import config_generation_with_cmd_args

config = None

if __name__ == "__main__":
    config = config_generation_with_cmd_args()
    automl_csv.config = config
    automl_csv.main()

    # time.sleep(60)

    automl_dataset.config = config
    automl_dataset.main()

    # time.sleep(60)

    n = 5
    for i in range(1, 1 + n):
        config.training_num = i
        automl_train.config = config
        automl_train.main()

        # time.sleep(60)
