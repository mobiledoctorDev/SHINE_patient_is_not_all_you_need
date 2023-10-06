import os, sys
import logging

from datetime import datetime
import traceback
from multiprocessing.pool import ThreadPool
import argparse

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, Table, Column, Integer, String, MetaData, DateTime, Float

from db_connectors import PostgresConnector, BaseDBConnector
from db_connectors import KT_SHINE_DB_YML_PATH
from calc_distance import distance_with_angle

TARGET_GPS_TABLE_NAME = 'shine2_pp_count_gps'
TARGET_BTS_TABLE_NAME = 'shine2_pp_count_bts'
RESULT_INDEX_FORMAT = '{pid}__{date}'
N_THREAD = 10

logger = logging.getLogger(__name__)


def main_gps(df_user):
    print(
        f"patient_id count:{df_user['patient_id'].nunique()}, "
        f"{df_user['patient_id'].min()} ~ {df_user['patient_id'].max()}"
    )
    for pid in df_user['patient_id'].unique():
        update_gps_table_for_patient(pid)


def main_gps_multithread(df_user):
    print(
        f"patient_id count:{df_user['patient_id'].nunique()}, "
        f"{df_user['patient_id'].min()} ~ {df_user['patient_id'].max()}"
    )
    pool = ThreadPool(N_THREAD)
    pool.map(update_gps_table_for_patient, df_user['patient_id'].unique())
    pool.close()
    pool.join()


def load_user_df(limit=None):
    # Set LIMIT expression
    limit_str = "" if limit is None else f"LIMIT {limit}"

    # Generate SQL query
    query = f"SELECT * FROM shine2_general_user {limit_str}"
    df = load_shine_data(query)
    logger.info("Load user data. shape:{}".format(df.shape))
    return df


def load_shine_data(query, engine=None):
    if engine is None:
        engine = BaseDBConnector(KT_SHINE_DB_YML_PATH, PostgresConnector).get_engine()

    df = pd.read_sql(query, engine)
    return df


def load_raw_gps_from_shine(pid):
    sql = f"SELECT * FROM public.shine2_general_gps WHERE patient_id={pid}"
    return load_shine_data(sql)


def apply_measure(row):
    """ pands apply function. row: [lat1, lon1, lat2, lon2]"""
    lat1, lon1, lat2, lon2 = row
    return distance_with_angle(lat1, lon1, lat2, lon2)


def update_gps_table_for_patient(pid, target_table=TARGET_GPS_TABLE_NAME):
    # 유저 한명의 gps 데이터를 불러온다.
    df_p_gps = load_raw_gps_from_shine(pid)

    if df_p_gps.empty:
        logger.debug(f"pid:{pid}, No GPS data. Skip run.")
        return
    else:
        logger.info(f"pid:{pid}, shape:{df_p_gps.shape}. Start run.")

    # 유저 한명의 gps 데이터를 bigquery에 업데이트한다.
    updater = GpsAverageProcessorToShine(target_table)
    updater.run(pid, df_p_gps)
    updater.close()


class GpsAverageProcessorToShine:

    def __init__(self, target_table, engine=None):
        """
        Calculate one day of a user's gps average, standard deviation and total moving distance and put into bigquery table.
        Args:
            engine: Google bigquery Client
            target_table (str): target table name
        """
        self.engine = engine if engine is not None else BaseDBConnector(KT_SHINE_DB_YML_PATH, PostgresConnector).get_engine()
        self.target_table = target_table
        self.results = []

    def run(self, pid, df_p_gps):
        # 날짜별로 각각 생성
        df_p_gps['date'] = df_p_gps['register_date'].astype(str).apply(lambda x: x[:10])

        for date, df_date in df_p_gps.groupby("date"):
            logger.debug(f"pid:{pid}, date:{date}, shape:{df_date.shape}")
            index = RESULT_INDEX_FORMAT.format(pid=pid, date=date)

            if self.is_exist(pid, date):
                logger.debug(f"index {index} is already exist in the bigquery table. Skip.")
                continue

            dict_new_info = generate_new_info(index, date, df_date)
            logger.debug(f"dict_new_info:{dict_new_info}")
            ret = self.insert_new_info(dict_new_info)
            self.results.append(ret)

        if self.results:
            logger.info(
                f"All success results: {sum(self.results)}, "
                f"total results:{len(self.results)}, "
                f"success rate:{sum(self.results) / len(self.results):.2f}%"
            )

    def is_exist(self, pid, date):
        index = RESULT_INDEX_FORMAT.format(pid=pid, date=date)
        sql = f"SELECT * FROM {self.target_table} WHERE index='{index}'"
        logger.debug(f" is_exist sql:{sql}")
        df = load_shine_data(sql, self.engine)
        return df.shape[0] > 0

    def insert_new_info(self, dict_new_info):

        conn = self.engine.connect()

        # Create metadata object
        metadata = MetaData()

        # Define table structure
        pp_table = Table('shine2_pp_count_gps', metadata,
                         Column('index', String, primary_key=True),
                         Column('n', Integer),
                         Column('long_mean', Float),
                         Column('lati_mean', Float),
                         Column('long_std', Float),
                         Column('lati_std', Float),
                         Column('path_sum', Float),
                         )

        # Create insert statement
        insert_stmt = pp_table.insert().values(**dict_new_info)

        # Execute the statement
        try:
            conn.execute(insert_stmt)
            logger.info(f"New rows have been added. index:{dict_new_info['index']}")
            return True
        except Exception as e:
            logger.exception(f"Encountered errors while inserting rows: {e}")
            return False

    def close(self):
        self.engine.dispose()


def generate_new_info(index, date, df_date):
    dict_new_info = dict()
    dict_new_info['index'] = index

    # 위도, 경도의 평균 및 표준편차 계산
    n = df_date.shape[0]
    dict_new_info['n'] = n
    dict_new_info['long_mean'] = df_date['longitude'].mean()
    dict_new_info['lati_mean'] = df_date['latitude'].mean()
    dict_new_info['long_std'] = df_date['longitude'].std() if n > 1 else None
    dict_new_info['lati_std'] = df_date['longitude'].std() if n > 1 else None

    # 총 이동거리 계산
    df_date = df_date.sort_values("register_date")
    df_date['rolling_lati'] = df_date['latitude'].shift(1)
    df_date['rolling_long'] = df_date['longitude'].shift(1)
    df_date['path_length'] = df_date[['latitude', 'longitude', 'rolling_lati', 'rolling_long']].apply(
        apply_measure, axis=1
    )
    dict_new_info['path_sum'] = df_date['path_length'].sum()

    return dict_new_info


def main_bts(today="2022-01-01"):
    connector = BaseDBConnector(KT_SHINE_DB_YML_PATH, PostgresConnector)
    engine = connector.get_engine()

    print("today:", today)

    df = load_user_df()
    print(f"patient_id - count:{df['patient_id'].nunique()}, {df['patient_id'].min()} ~ {df['patient_id'].max()}")

    df_count_bts_date = pd.DataFrame()

    i_count = 0

    for pid in sorted(df['patient_id'].unique()):

        sql = f"SELECT * FROM public.shine2_general_bts WHERE patient_id={pid}"
        df_p_bts = load_shine_data(sql, engine)

        print(f"pid:{pid}", df_p_bts.shape)

        df_p_bts['date'] = df_p_bts['bts_start_date'].astype(str).apply(lambda x: x[:10])
        for date, df_date in df_p_bts.groupby("date"):
            print(f"pid:{pid}, date:{date}, shape:{df_date.shape}")
            index = f"{pid}__{date}"

            # 위도, 경도의 평균 및 표준편차
            df_count_bts_date.loc[index, 'n'] = df_date.shape[0]
            df_count_bts_date.loc[index, 'long_mean'] = df_date['longitude'].mean()
            df_count_bts_date.loc[index, 'lati_mean'] = df_date['latitude'].mean()
            df_count_bts_date.loc[index, 'long_std'] = df_date['longitude'].std()
            df_count_bts_date.loc[index, 'lati_std'] = df_date['latitude'].std()

            # 총 이동거리
            df_date = df_date.sort_values("bts_start_date")
            df_date['rolling_lati'] = df_date['latitude'].shift(1)
            df_date['rolling_long'] = df_date['longitude'].shift(1)
            df_date['path_length'] = df_date[['latitude', 'longitude', 'rolling_lati', 'rolling_long']].apply(
                apply_measure, axis=1
            )
            df_count_bts_date.loc[index, 'path_sum'] = df_date['path_length'].sum()

            i_count += 1

            if i_count % 1000 == 0:
                print("icount:", i_count, "save_temp df_count_bts_date:", df_count_bts_date.shape)
                df_count_bts_date.to_csv(f'./location_data/tmp_bts_count_{today}.csv', index=True)

    df_count_bts_date.to_csv(f'./location_data/final_bts_count_{today}.csv', index=True)


if __name__ == "__main__":
    # # main_bts("2022-01-01")
    # logging.basicConfig(level=logging.INFO, format='(%(asctime)s) [%(threadName)s] %(levelname)s: %(message)s')
    # logger = logging.getLogger(__name__)
    #
    # # Argument Parsing
    # parser = argparse.ArgumentParser(description='Process some integers.')
    # parser.add_argument('--table_name', type=str, default=TARGET_TABLE_NAME,
    #                     help='name of the database table')
    # parser.add_argument('--result_index_format', type=str, default=RESULT_INDEX_FORMAT,
    #                     help='format string for result index')
    # parser.add_argument('--n_thread', type=int, default=N_THREAD,
    #                     help='number of threads to use')
    # args = parser.parse_args()
    #
    # TARGET_TABLE_NAME = args.table_name
    # RESULT_INDEX_FORMAT = args.result_index_format
    # N_THREAD = args.n_thread
    #
    # # main run
    # df_user = load_user_df()
    # main_gps_multithread(df_user)
    # """테스트 부분은 test_postprocessor_loc에 작성"""

    main_bts(today="2023-04-01")