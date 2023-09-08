import datetime
import os
import pandas as pd
import logging

from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, Table, Column, Integer, String, MetaData, DateTime, Float

from post_processor_loc_shine import load_shine_data, TARGET_GPS_TABLE_NAME

logger = logging.getLogger(__name__)

SAVE_DIR = './tmp_output'


def run_whole_data_to_csv(limit=None):
    try:
        # Set LIMIT expression
        limit_str = "" if limit is None else f"LIMIT {limit}"

        # Generate SQL query
        query = f"SELECT * FROM {TARGET_GPS_TABLE_NAME} {limit_str}"
        df = load_shine_data(query)
        logger.info("Load user data. shape:{}".format(df.shape))

        if not os.path.exists(SAVE_DIR):
            os.makedirs(SAVE_DIR)
        df.to_csv(f'{SAVE_DIR}/gps_full_{datetime.datetime.now().strftime("%Y-%m-%d")}.csv')

    except Exception as e:
        logger.exception(f"Error occurred while loading data from DB Error:{e}")
        return False

    return True

# Don't forget save BTS csv file into same directory.

if __name__ == '__main__':
    run_whole_data_to_csv()