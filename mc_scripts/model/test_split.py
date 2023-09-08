import pandas as pd
from benchmarks import split_by_patientid, data_split


def test_split_by_patientid():
    df_all = pd.read_csv("../../pipeline/resource/shine_v3_3_kt/df_train_v3_3_kt.csv")
    df_trainval, df_test = split_by_patientid(df_all)
    assert df_trainval.shape[0] + df_test.shape[0] == df_all.shape[0]

    train_patientids = set(df_trainval['patient_id'].unique())
    test_patientids = set(df_test['patient_id'].unique())

    assert len(train_patientids.intersection(test_patientids)) == 0


def test_split_manual():
    df_all = pd.read_csv("../../pipeline/output/shine_v3_3_kt/df_all.csv")
    df_train, df_val, df_test = data_split(df_all, how='manual')
    assert df_train.shape[0] + df_val.shape[0] + df_test.shape[0] == df_all.shape[0]

    train_patientids = set(df_train['patient_id'].unique())
    val_patientids = set(df_val['patient_id'].unique())
    test_patientids = set(df_test['patient_id'].unique())

    assert len(train_patientids.intersection(test_patientids)) == 0
    assert len(train_patientids.intersection(val_patientids)) == 0
    assert len(val_patientids.intersection(test_patientids)) == 0

    assert all(df_train['split'] == ['TRAIN'] * df_train.shape[0])
    assert all(df_val['split'] == ['VALIDATE'] * df_val.shape[0])
    assert all(df_test['split'] == ['TEST'] * df_test.shape[0])


def test_split_testmonthvalrandom():
    df_all = pd.read_csv("../../pipeline/resource/shine_v3_3_kt/df_train_v3_3_kt.csv")
    df_train, df_val, df_test = data_split(df_all, how='testmonthvalrandom', test_month='2022-09')

    train_patientids = set(df_train['patient_id'].unique())
    val_patientids = set(df_val['patient_id'].unique())
    assert len(train_patientids.intersection(val_patientids)) == 0
    assert df_test['selfcheck_date'].min() >= '2022-09-01'
    assert "2022-10-01" not in df_test['selfcheck_date']
    assert "2022-10-01" not in df_train['selfcheck_date']
    assert "2022-09-01" not in df_train['selfcheck_date']
    assert "2022-09-01" not in df_val['selfcheck_date']
    assert not df_train.empty
    assert not df_val.empty
    assert not df_test.empty

