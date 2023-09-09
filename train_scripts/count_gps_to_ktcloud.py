import os, sys
import logging

from datetime import datetime
import traceback

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from pipeline.loc_data_postprocessing.db_connectors import PostgresConnector, BaseDBConnector
from pipeline.loc_data_postprocessing.db_connectors import KT_SHINE_DB_YML_PATH


def main():
    df_count_gps = pd.read_csv(f"{os.path.dirname(__file__)}/pp_data/gps_count_gc.csv")
    engine = BaseDBConnector(KT_SHINE_DB_YML_PATH, PostgresConnector).get_engine()
    df_count_gps.to_sql('shine2_pp_count_gps', engine, if_exists='replace', index=False)


if __name__ == "__main__":
    main()
