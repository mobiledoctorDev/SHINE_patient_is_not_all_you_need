import os
import datetime
import logging
import pytest
import pandas as pd

from post_processor_loc_shine import load_user_df, apply_measure, load_shine_data, load_raw_gps_from_shine, \
    GpsAverageProcessorToShine, generate_new_info

logger = logging.getLogger(__name__)

TEST_TABLE_NAME = 'shine2_pp_count_gps_test'

def test_load_shine_data():
    df_result = load_shine_data("SELECT 1+1 as two;")
    logger.info(f"result:\n{str(df_result)}")
    assert not df_result.empty


def test_user_df_selector():
    df_user = load_user_df(limit=100)
    logger.info(f"df_user.shape:{df_user.shape}, df_user.columns:{df_user.columns.tolist()}")
    assert df_user.shape[0] == 100
    assert "patient_id" in df_user.columns
    assert "birthday" in df_user.columns
    assert "address" in df_user.columns
    assert "join_date" in df_user.columns
    assert "carrier" in df_user.columns
    assert "subs_path" in df_user.columns
    assert "promo_code" in df_user.columns


def test_df_gps_from_shine():
    # 유저 한명의 gps 데이터를 불러온다.
    pid = 53532
    df_p_gps = load_raw_gps_from_shine(pid)
    logger.info(f"df_p_gps.shape:{df_p_gps.shape}, df_p_gps.columns:{df_p_gps.columns.tolist()}")
    assert df_p_gps.shape[0] > 0
    assert "patient_id" in df_p_gps.columns.tolist()


def test_df_gps_is_exist():
    """is_exist 함수를 테스트합니다. """
    updater = GpsAverageProcessorToShine(TEST_TABLE_NAME)
    ret = updater.is_exist(47764, "2022-03-30")
    updater.close()
    assert ret, f"Ret: {ret}"


def test_insert():
    test_info = {
        'index': "123__2020-01-01", 'n': 505,
        'long_mean': 127.16099747653502, 'lati_mean': 36.902082331712705,
        'long_std': 7.796249230452145e-05, 'lati_std': 5.902354700505511e-05,
        'path_sum': 608.0559507108607
    }

    updater = GpsAverageProcessorToShine(TEST_TABLE_NAME)
    ret = updater.insert_new_info(dict_new_info=test_info)
    updater.close()
    assert ret


def test_insert_nan():
    test_info = {
        'index': "123__2020-01-01", 'n': 1,
        'long_mean': 127.16099747653502, 'lati_mean': 36.902082331712705,
        'long_std': None, 'lati_std': None,
        'path_sum': 0.0
    }

    updater = GpsAverageProcessorToShine('shine-mobiledodctor.shine_20230101.tmp_test_gps')
    ret = updater.insert_new_info(dict_new_info=test_info)
    updater.close()
    assert ret


def test_generate_info():
    index = '111__2020-01-02'
    date = '2020-01-02'
    df_date = pd.DataFrame({
        'register_date': [datetime.datetime.fromisoformat('2020-01-02 00:00:00')],
        'longitude': [127.16099747653502],
        'latitude': [36.902082331712705],
    })

    dict_info = generate_new_info(index, date, df_date)
    logger.info(dict_info)
    assert dict_info['index'] == "111__2020-01-02"
    assert dict_info['long_mean'] == pytest.approx(127.16099747653502)
    assert dict_info['lati_mean'] == pytest.approx(36.902082331712705)
    assert dict_info['path_sum'] == pytest.approx(0.0)
    assert dict_info['n'] == 1
    assert dict_info['long_std'] is None
    assert dict_info['lati_std'] is None
