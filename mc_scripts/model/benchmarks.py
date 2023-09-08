import os
import logging

import datetime
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import argparse

from sklearn.metrics import roc_auc_score, confusion_matrix, precision_recall_curve, average_precision_score
# from sklearn import svm
# from tqdm.notebook import tqdm
import pickle

import matplotlib.pyplot as plt
from features import dict_set_of_cols

config = None
SYSTEM_COLS = ['patient_id', 'split']  # 학습을 돌리는데 시스템에서 필요한 정보들입니다. 학습에 사용되면 안됩니다.
GT_COL = 'pcr_result'
GT_COLS = [GT_COL]

TRAIN_TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')


def data_load(config):
    bucket_name = config.bucket_name
    filepath = f"../../pipeline/output/{bucket_name}/df_all_added_owid_loc.csv"
    df_all = pd.read_csv(filepath)

    if config.onlyuse_selfcheck_first:
        df_all = df_all[df_all['pcr_date'] >= df_all['selfcheck_date']]
    return df_all


def get_train_cols(target_set_string='patient+si6'):
    if target_set_string == 'all':
        target_sets = dict_set_of_cols.keys()
    else:
        target_sets = target_set_string.split("+")

    train_cols = []
    for target_set in target_sets:
        train_cols += dict_set_of_cols[target_set]
    print(train_cols)
    train_cols = remove_duplicates(train_cols)

    if not train_cols:
        raise ValueError(f'No columns are selected. target_sets:{target_sets}')
    return train_cols


def remove_duplicates(list_cols):
    """
    This function removes duplicate elements from a list while maintaining the order of elements
    and keeping the first occurrence of each element.
    """
    seen = set()
    result = []
    for item in list_cols:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def filter_target_columns(df_all, col_string='si6', fill_null=True):
    base_columns = SYSTEM_COLS + GT_COLS
    train_cols = get_train_cols(col_string)
    print('train_cols:', train_cols)
    base_columns += train_cols
    df_all = df_all[base_columns].copy()
    print("before fill null:", df_all.isnull().sum())

    if fill_null:
        df_all = df_all.fillna(0)
        print("after fill Null:", df_all.isnull().sum())
    else:
        cols_mask = [x for x in df_all.columns if 'mask' in x]
        for mask_col in cols_mask:
            if mask_col == 'sigungu_confirmed_mask':
                val_col = 'sigungu_confirmed_ratio'
                df_all.loc[df_all[val_col] == 0, val_col] = None
            else:
                val_col = mask_col.replace('mask', 'norm')
                df_all.loc[df_all[mask_col] == 1, val_col] = None

        cols_without_mask = [x for x in df_all.columns if 'mask' not in x]
        df_all = df_all[cols_without_mask]
        df_all = df_all.fillna(-1)
        print("Not fill Null:", df_all.isnull().sum())

    return df_all


def data_split_by_month_last_month_for_test(df, test_month, date_col='selfcheck_date'):
    # Filter out data from the later target test month
    df_all = df[df[date_col] <= f'{test_month}-31']
    # Get df_test from the last month
    df_test = df_all[df_all[date_col] >= f'{test_month}-01']
    df_trainval = df_all[df_all[date_col] < f'{test_month}-01']
    return df_trainval, df_test


def data_split(df_all, how, test_month=None, val_month=None):
    if how == 'manual':
        assert test_month is None
        df_train = df_all[df_all['split'] == 'TRAIN']
        df_val = df_all[df_all['split'] == 'VALIDATE']
        df_test = df_all[df_all['split'] == 'TEST']
    elif how == 'testmonthvalrandom':
        assert test_month is not None
        df_trainval, df_test = data_split_by_month_last_month_for_test(df_all, test_month)
        df_train, df_val = split_by_patientid(df_trainval, test_size=0.2, random_state=42)
    elif how == 'testmonthvalmonth':
        assert val_month is not None
        assert test_month is not None
        df_trainval, df_test = data_split_by_month_last_month_for_test(df_all, test_month)
        df_train, df_val = data_split_by_month_last_month_for_test(df_trainval, val_month)
    else:
        raise NotImplementedError(f"Unknown split method: {how}")

    print(df_train.shape, df_val.shape, df_test.shape)
    return df_train, df_val, df_test


def split_by_patientid(df, test_size=0.2, random_state=1212):
    patient_id = df['patient_id'].unique()
    # Select test_size of patient_id
    np.random.seed(random_state)
    patient_id_test = np.random.choice(patient_id, size=int(len(patient_id) * test_size), replace=False)
    df_test = df[df['patient_id'].isin(patient_id_test)]
    df_train = df[~df['patient_id'].isin(patient_id_test)]
    return df_train, df_test


def remove_patient_id_and_result(df):
    exclude_cols = SYSTEM_COLS + GT_COLS
    cols = [x for x in df.columns if x not in exclude_cols]
    return df[cols], df[GT_COLS]


def run_one_full_train_test(df_trainval, df_test, cols, model):
    x = df_trainval[cols]
    y = df_trainval[GT_COL]
    print(f"start to train: x.shape:{x.shape}, y.shape:{y.shape}")

    model.fit(x, y)
    print(f"Model: {model}")
    pred_pos = get_pred_pos(df_trainval[cols], model)
    gt_train = df_trainval[GT_COL]

    print(pred_pos.shape, gt_train.shape)
    print("train set score", calc_eval_score(gt_train, pred_pos))

    pred_pos = get_pred_pos(df_test[cols], model)
    gt_test = df_test[GT_COL]
    test_result = calc_eval_score(gt_test, pred_pos)
    print("test set score", test_result)

    return model, test_result


def run_model_with_random_5_trainsets(df_trainval, df_test, model, model_name):
    train_cols = get_train_cols(config.using_features)
    test_result = dict()
    models = dict()
    for i in range(5):
        models[i], test_result[i] = run_one_full_train_test(
            df_trainval.sample(int(df_trainval.shape[0] * 0.8), random_state=config.seed + i),
            df_test, train_cols,
            model
        )
        model_filepath = f"{get_save_dir()}/model_{model_name}_{i}.pkl"
        pickle.dump(model, open(model_filepath, 'wb'))
        print(f"Sucessfully save the model to :{model_filepath}")

    test_result = calc_mean_and_std_from_result(test_result)
    print(f"[{model_name}]", test_result)
    return test_result


def calc_mean_and_std_from_result(test_result):
    print("values:", list(test_result.values()))
    test_result['mean'] = dict()
    for key in test_result[0].keys():
        test_result['mean'][key] = np.mean([test_result[i][key] for i in range(5)])

    test_result['std'] = dict()
    for key in test_result[0].keys():
        test_result['std'][key] = np.std([test_result[i][key] for i in range(5)])
    return test_result


def run_lr(df_trainval, df_test):
    from sklearn.linear_model import LogisticRegression

    lr_model = LogisticRegression()
    test_result = run_model_with_random_5_trainsets(df_trainval, df_test, lr_model, "LR")

    return test_result


def calc_eval_score(gt_train, pred_pos):
    dict_score = dict()
    dict_score['auroc'] = roc_auc_score(gt_train, pred_pos)
    dict_score['average_precision'] = average_precision_score(gt_train, pred_pos)
    precision, recall, thresholds = precision_recall_curve(gt_train, pred_pos)
    f1_scores = (2 * precision * recall) / (precision + recall)
    threshold = thresholds[f1_scores[:-1].argmax()]
    print("threshold:", threshold)
    binary_preds = [1 if p >= threshold else 0 for p in pred_pos]
    # Get the confusion matrix
    tn, fp, fn, tp = confusion_matrix(gt_train, binary_preds).ravel()
    dict_score['threshold'] = threshold
    dict_score['precision'] = precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    dict_score['recall'] = recall = tp / (tp + fn) if (tp + fn) != 0 else 0
    dict_score['sensitivity'] = sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0
    dict_score['specificity'] = specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
    dict_score['f1'] = f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    return dict_score


def run_one_full_train_test_with_validation_set(df_train, df_val, df_test, cols, model):
    x_train = df_train[cols].to_numpy()
    y_train = df_train['pcr_result'].to_numpy()
    x_val = df_val[cols].to_numpy()
    y_val = df_val['pcr_result'].to_numpy()

    model.fit(
        X=x_train, y=y_train,
        eval_set=[(x_val, y_val)],
        eval_metric=['auc'],
    )
    print(f"Model: {model}")

    input_train = df_train[cols]
    pred_pos = get_pred_pos(input_train, model)
    gt_train = df_train[GT_COL]
    print(pred_pos.shape, gt_train.shape)
    print("train set score", calc_eval_score(gt_train, pred_pos))

    input_test = df_test[cols]
    pred_pos = get_pred_pos(input_test, model)
    gt_test = df_test[GT_COL]
    test_result = calc_eval_score(gt_test, pred_pos)
    print("test set acore", test_result)

    return model, test_result


def get_pred_pos(df_input, model):
    if getattr(model, 'predict_proba', False):
        print("Found predict_proba")
        pred_raw = model.predict_proba(df_input)
        pred_pos = np.array([x[1] for x in pred_raw])
    else:
        pred_pos = model.predict(df_input)
    return pred_pos


def run_xgboost(df_trainval, df_test):
    import xgboost as xgb

    train_cols = get_train_cols(config.using_features)
    test_result = dict()
    models = dict()
    for i in range(5):
        xg_reg = xgb.XGBRegressor(objective='reg:linear', n_estimators=10, random_state=i + config.seed)
        df_train, df_val = split_by_patientid(df_trainval, test_size=0.2, random_state=i + config.seed)
        print(f"Split trainval for XgBoost "
              f"with seed:{i + config.seed} trainshape:{df_train.shape} valshape:{df_val.shape}")

        models[i], test_result[i] = run_one_full_train_test_with_validation_set(
            df_train, df_val, df_test, train_cols,
            xg_reg
        )
        model_filepath = f"{get_save_dir()}/model_xgboost_{i}.pkl"
        pickle.dump(xg_reg, open(model_filepath, 'wb'))
        print(f"Sucessfully save the model to :{model_filepath}")

    test_result = calc_mean_and_std_from_result(test_result)
    print(f"[XgBoost]", test_result)
    return test_result


def run_lgbm(df_trainval, df_test):
    import lightgbm as lgb

    train_cols = get_train_cols(config.using_features)
    test_result = dict()
    models = dict()
    for i in range(5):
        lgbm_model = lgb.LGBMClassifier(random_state=i + config.seed)
        df_train, df_val = split_by_patientid(df_trainval, test_size=0.2, random_state=i + config.seed)
        print(f"Split trainval for LGBM "
              f"with seed:{i + config.seed} trainshape:{df_train.shape} valshape:{df_val.shape}")
        models[i], test_result[i] = run_one_full_train_test_with_validation_set(
            df_train, df_val, df_test, train_cols,
            lgbm_model
        )
        model_filepath = f"{get_save_dir()}/model_lgbm_{i}.pkl"
        pickle.dump(lgbm_model, open(model_filepath, 'wb'))
        print(f"Sucessfully save the model to :{model_filepath}")

    test_result = calc_mean_and_std_from_result(test_result)
    print(f"[LGBM]", test_result)
    return test_result


def run_tabnet(df_trainval, df_test):
    import pandas as pd
    import numpy as np
    from pytorch_tabnet.tab_model import TabNetClassifier

    def run_one_full_train_test_with_tabnet(df_train, df_val, df_test, cols, model):
        x_train = df_train[cols].to_numpy()
        y_train = df_train['pcr_result'].to_numpy()
        x_val = df_val[cols].to_numpy()
        y_val = df_val['pcr_result'].to_numpy()
        x_test = df_test[cols].to_numpy()
        y_test = df_test['pcr_result'].to_numpy()

        print(f"start to train: x_train.shape:{x_train.shape}, y_train.shape:{y_train.shape}")
        print(f"start to validation: x_val.shape:{x_val.shape}, y_val.shape:{y_val.shape}")
        print(f"y_train mean:{np.mean(y_train)}. y_val mean:{np.mean(y_val)}")

        model.fit(
            X_train=x_train, y_train=y_train,
            eval_set=[(x_train, y_train), (x_val, y_val)],
            eval_name=['train', 'valid'],
            patience=100,
            max_epochs=1000,
            eval_metric=['auc'],
        )

        print(f"Model: {model}")
        # plot losses
        plt.plot(model.history['loss'], label='loss')

        # plot accuracy
        plt.plot(model.history['train_auc'], label='train_auc')
        plt.plot(model.history['valid_auc'], label='valid_auc')
        plt.legend(['loss', 'train_auc', 'valid_auc'])
        plt.savefig(f"{get_save_dir()}/model_history_TabNet_{i}.png")
        plt.clf()

        # determine best accuracy for validation set
        preds_valid = model.predict_proba(x_val)
        print(f"y_val.head:{y_val[:5]}")
        pred_pos = np.array([x[1] for x in preds_valid])
        valid_score = calc_eval_score(y_val, pred_pos)
        print(f"BEST SCORE ON VALIDATION SET : {valid_score}")

        print(f"start to test: x_test.shape:{x_test.shape}, y_test.shape:{y_test.shape}")
        # find and plot feature importance
        y_pred = model.predict(x_test)
        feature_importances_ = model.feature_importances_
        print("feature_importance:", feature_importances_)
        feat_importances = pd.Series(model.feature_importances_, index=cols)
        feat_importances.nlargest(20).plot(kind='barh')
        plt.savefig(f"{get_save_dir()}/feature_importance_TabNet_{i}.png")
        plt.clf()

        # determine best accuracy for test set
        preds = model.predict_proba(x_test)
        pred_pos = np.array([x[1] for x in preds])
        test_score = calc_eval_score(y_test, pred_pos)
        print(f"preds_test head:{preds[:5]}")
        print(f"BEST SCORE ON TEST SET : {test_score}")

        config_name = 'TabNet_DEFAULT_best'
        best_epoch, element = max(enumerate(model.history['valid_auc']), key=lambda x: x[1])
        print(f"Best epoch:{best_epoch}")
        best_model = save_model(config_name, model, best_epoch)

        return model, test_score

    def save_model(model_name, model, epoch):
        save_path = f"./{get_save_dir()}/{model.__class__.__name__}_{model_name}_{epoch:04d}.pth"
        print("save_path", save_path)
        model.save_model(save_path)
        print("Success to save to : ", save_path)
        return model

    test_result = dict()
    models = dict()
    train_cols = get_train_cols(target_set_string=config.using_features)
    for i in range(5):
        tab_model = TabNetClassifier(verbose=2, seed=config.seed + i)
        df_train, df_val = split_by_patientid(df_trainval, test_size=0.2, random_state=i + config.seed)
        print(
            f"Split trainval for TabNet with seed:{i + config.seed} trainshape:{df_train.shape} valshape:{df_val.shape}")
        models[i], test_result[i] = run_one_full_train_test_with_tabnet(
            df_train, df_val, df_test,
            train_cols, tab_model
        )

    test_result = calc_mean_and_std_from_result(test_result)
    print(f"[TabNet]", test_result)
    return test_result


def main(config_from_cmd):
    global config
    config = config_from_cmd
    split_type = config.split

    # Load data
    df_all = data_load(config)
    df_train, df_val, df_test = data_split(df_all, how=split_type, test_month=config.test_month,
                                           val_month=config.val_month)
    df_train = filter_target_columns(df_train, config.using_features)
    df_val = filter_target_columns(df_val, config.using_features)
    df_test = filter_target_columns(df_test, config.using_features)

    df_trainval = pd.concat([df_train, df_val], axis=0)
    print(f"df_trainval.shape:{df_trainval.shape}, df_test.shape:{df_test.shape}")
    print(f"df_trainval patient_id:", df_trainval['patient_id'].nunique(), df_test['patient_id'].nunique())

    # Run!
    final_result = dict()
    final_result['LR'] = run_lr(df_trainval, df_test)
    save_temporary_result(final_result)

    print("[3Split]Train:", df_train.shape, df_train['patient_id'].nunique())
    print("[3Split]Validaiton:", df_val.shape, df_val['patient_id'].nunique())
    print("[3Split]Test:", df_test.shape, df_test['patient_id'].nunique())

    final_result['XGBoost'] = run_xgboost(df_trainval, df_test)
    save_temporary_result(final_result)
    final_result['LGBM'] = run_lgbm(df_trainval, df_test)
    save_temporary_result(final_result)
    final_result['TabNet'] = run_tabnet(df_trainval, df_test)
    save_temporary_result(final_result)

    print(final_result)
    return final_result


def get_save_dir():
    dataset_name = get_dataset_name(config)
    save_dir = f"./train_result/{config.version}_{TRAIN_TIMESTAMP}_{dataset_name}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    return save_dir


def get_dataset_name(config):
    test_month_str = "" if config.test_month is None else f"_test{config.test_month}"
    val_month_str = "" if config.val_month is None else f"_val{config.val_month}"
    onlyuse_selfcheck_first_str = '' if config.onlyuse_selfcheck_first is False else f'_onlyscfirst{config.onlyuse_selfcheck_first}'
    fillna_str = '_fillna' if config.fill_null else '_nofillna'
    return f'{config.using_features}_{config.split}' \
           f'{test_month_str}{val_month_str}{onlyuse_selfcheck_first_str}{fillna_str}'


def save_temporary_result(final_result):
    """Save dictionary as csv file"""
    save_dir = get_save_dir()
    # convert dictionary to dataframe
    print("Final _result:", final_result)
    dfs = []
    for model_key in final_result:
        model_result = final_result[model_key]
        df_model = pd.DataFrame.from_dict(model_result, orient='columns')
        df_model['model'] = model_key
        df_model = df_model.reset_index()
        df_model = df_model.set_index(['model', 'index'])
        dfs.append(df_model)
    df = pd.concat(dfs, axis=0)
    save_path = f"{save_dir}/result.csv"
    df.to_csv(save_path)


def config_generation_with_cmd_args():
    parser = argparse.ArgumentParser()
    # FILTER_VERSION can have 3 options: patient, si6, owid
    parser.add_argument('--using_features', type=str, default='patient',
                        help='You can sett this columns with + symbol. Something like, '
                             f'Options are: {dict_set_of_cols.keys()}')
    parser.add_argument('--split', type=str, default='manual',
                        help='split can have 3 options: manual, random, timestamp')
    parser.add_argument('--seed', type=int, default=1212,
                        help='seed for random split')
    parser.add_argument('--bucket_name', type=str)
    parser.add_argument('--test_month', type=str, help='test month. the format is YYYY-MM')
    parser.add_argument('--val_month', type=str, help='val month. the format is YYYY-MM')
    parser.add_argument('--onlyuse_selfcheck_first', action='store_true',
                        help='only use selfcheck date comes before PCR date or same date')
    parser.add_argument('--fill_null', action='store_true', help='fill null with zero')
    parser.add_argument('--training_num', type=int, default=None,
                        help="This will be used when to repeat training with same configure.")

    config = parser.parse_args()

    # split 날짜로 하는경우에는 반드시 test_month가 지정이 되어 있어야함.
    assert config.split == 'manual' or config.test_month is not None

    # Add some custom config for the model
    config.version = 'v22'

    return config


if __name__ == '__main__':
    config = config_generation_with_cmd_args()
    main(config)
