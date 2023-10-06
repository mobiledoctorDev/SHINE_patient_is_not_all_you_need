import pandas as pd
import numpy as np
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def apply_and_concat(df, field, func, cols_name):
    return pd.concat((df, df[field].apply(lambda cell: pd.Series(func(cell), index=cols_name))), axis=1)

def generate_diff_date(selfcheck_date, pcr_date):
    selfcheck_date = pd.to_datetime(selfcheck_date, format='%Y-%m-%d')
    pcr_date = pd.to_datetime(pcr_date, format='%Y-%m-%d')
    delta = abs(int((selfcheck_date - pcr_date) / pd.Timedelta(days=1)))
    return delta

def commanum_revert_to_int(s):
    try:
        if len(s) >=4 and s[1] == ',':
            return int(s[:1] + s[2:])
        if s[-1] == '+':
            return int(s[:-2])
        return int(s)
    except ValueError:
        print(s)
        return s

def display_performance(clf, pred, valid_input, valid_gt):
    pred = np.array(pred)

    print('accuracy %.4f' % (accuracy_score(pred, valid_gt)))
    print('precision %.4f' % (precision_score(pred, valid_gt)))
    print('recall %.4f' % (recall_score(pred, valid_gt)))
    print('f1-score %.4f' % (f1_score(pred, valid_gt)))
    print(confusion_matrix(pred, valid_gt))
    plot_roc_curve(clf, valid_input, valid_gt)

def split_input_gt(non_split_data):
    splited_input = non_split_data[:, :-1]
    splited_gt = non_split_data[:, -1]
    
    return splited_input, splited_gt

def split_train_test(data, test_ratio):
#     shuffled_indices = np.random.permutation(len(data)).astype(int)
    test_set_size = int(len(data) * test_ratio)
    
#     test_indices = shuffled_indices[:test_set_size]
#     train_indices = shuffled_indices[test_set_size:]
    
    print(len(data))
#     print(test_indices.dtype)
#     print(train_indices.dtype)
    
    test_data = data[:test_set_size]
    train_data = data[test_set_size:]
#     test_data = data[test_indices]
#     train_data = data[train_indices]
    
    return train_data, test_data

def split_df_with_ratio(df, ratio=0.2):
    baby_ids = df['patient_id'].unique()
    np.random.shuffle(baby_ids)
    test_set_size = int(len(baby_ids) * ratio)
    test_ids = baby_ids[:test_set_size]
    train_ids = baby_ids[test_set_size:]
    print(f'baby_ids가 나눠진 갯수 {len(train_ids)} {len(test_ids)}')
    return df[df['patient_id'].isin(train_ids)], df[df['patient_id'].isin(test_ids)]