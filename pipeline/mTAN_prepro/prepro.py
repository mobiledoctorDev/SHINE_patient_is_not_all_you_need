# %% 
import itertools
import operator
import os
import random
import statistics
import tarfile
import time
import cProfile
from datetime import datetime, timedelta

import warnings
from pathlib import Path

import numpy as np
import numexpr as ne
import pandas as pd
import shutil
import pickle

from tqdm.auto import tqdm
from sklearn.preprocessing import MultiLabelBinarizer

# Disable chained assignment warning
pd.options.mode.chained_assignment = None
warnings.filterwarnings("ignore", category=UserWarning)

tqdm.pandas()

# %% 
# Global variable to control whether the timer_decorator is enabled or disabled
TIMER_ENABLED = True

def timer_decorator(func):
    def wrapper(*args, **kwargs):
        if TIMER_ENABLED:
            start_time = time.time()
        result = func(*args, **kwargs)
        if TIMER_ENABLED:
            end_time = time.time()
            print(f"Function {func.__name__} took {end_time - start_time:.5f} seconds to execute.")
        return result
    return wrapper


# %% 
@timer_decorator
def read_csv_files(paths):
    """
    Reads multiple CSV files from a list of paths and returns a list of dataframes.
    """
    files = [pd.read_csv(os.path.join(path, file)) for path in paths for file in os.listdir(path)]
    return files

@timer_decorator
def select_reference_data(ref_file_path, selected_cols=None):
    """
    Loads a reference CSV file and selects specific columns for Korea (KOR).
    Returns a dataframe containing the selected data.
    """
    with open(ref_file_path, 'r') as f:
        ref = pd.read_csv(f)
    if selected_cols is None:
        selected_cols = [
            'new_cases_per_million', 'reproduction_rate',
            'weekly_hosp_admissions_per_million',
            'people_fully_vaccinated_per_hundred', 'positive_rate']
    ref_kr = ref.loc[ref['iso_code'] == 'KOR', ['date'] + selected_cols].reset_index(drop=True)
    return ref_kr

# %%
@timer_decorator
def create_folders():
    for folder in ['./train_fevercoach', './test_fevercoach', './pos', './neg']:
        if os.path.exists(folder):
            print(f"Removing existing folder: {folder}")
            shutil.rmtree(folder)
        os.makedirs(folder)

# %% 
@timer_decorator
def drop_n_to_dt(df, columns_to_remove):
    df = df.drop(columns_to_remove, axis=1)
    df['selfcheck_date'] = pd.to_datetime(df['selfcheck_date'])
    return df

@timer_decorator
def remove_pregnant_rogues(selfcheck):
    pregnant_roguestr = [41100, 68852, 33368, 33363, 20228, 54455, 40001]
    selfcheck = selfcheck[~selfcheck['patient_id'].isin(pregnant_roguestr)]
    selfcheck['pregnant'] = selfcheck['pregnant'].astype(str).apply(
        lambda x: x[0] if '주 ' in x else x[:-1] if '주' in x else x.split('.')[0] if '.' in x else x)
    return selfcheck

@timer_decorator
def remove_bodytemp_rogues(selfcheck):
    selfcheck['fever_temp'] = pd.to_numeric(selfcheck['fever_temp'], errors='coerce')
    selfcheck['fever_temp'].replace({98.9: 37.4, 48.1: 38.1}, inplace=True)
    selfcheck['fever_temp'] = selfcheck['fever_temp'].apply(lambda x: x / 100 if x > 3000 else x / 10 if x > 300 else x if x > 30 else np.nan)
    return selfcheck

@timer_decorator
def create_dummy_variables_v3(df, col_name, prefix='scReason'):
    # Replace null values with empty string
    df[col_name] = df[col_name].fillna('')
    # Split column into lists (handle single-value strings)
    split_col = df[col_name].str.split('`')
    
    # Create MultiLabelBinarizer object
    mlb = MultiLabelBinarizer()
    # Fit and transform split_col
    dummy_cols = pd.DataFrame(mlb.fit_transform(split_col), columns=mlb.classes_)
    # Add prefix to column names
    dummy_cols = dummy_cols.add_prefix(prefix)#.replace(0, np.nan)
    if prefix in dummy_cols.columns:
        dummy_cols = dummy_cols.drop(prefix, axis=1)

    # Concatenate original dataframe with dummy variable columns
    df = pd.concat([df, dummy_cols], axis=1)
    # Drop original column
    df = df.drop(col_name, axis=1)
    return df

# %%
@timer_decorator
def processing_others(user_df, pcr_df, underlingdz_df):
    # Merge dataframes
    merged = user_df.merge(pcr_df, on='patient_id', how='left').merge(underlingdz_df, on='patient_id', how='left')

    # Convert birthday to float and drop rows with missing values in birthday or gender
    merged['birthday'] = pd.to_numeric(merged['birthday'], errors='coerce')
    merged.dropna(subset=['birthday', 'gender'], inplace=True)
    
    return merged


@timer_decorator
def get_with_pcr_chunks(selfcheck, pcr, percentage=0.2):
    with_pcr = selfcheck[selfcheck['patient_id'].isin(pcr['patient_id'].unique())].copy()
    testset = np.random.choice(with_pcr['patient_id'].unique(), int(len(with_pcr['patient_id'].unique())*percentage), replace=False)
    chunk1 = with_pcr[~with_pcr['patient_id'].isin(testset)].copy()
    chunk2 = with_pcr[with_pcr['patient_id'].isin(testset)].copy()
    return chunk1, chunk2

@timer_decorator
def get_chunks(selfcheck, pcr, percentage=0.2, with_pcr=True):
    pcr_ids = pcr['patient_id'].unique()
    with_pcr_mask = selfcheck['patient_id'].isin(pcr_ids)
    without_pcr_mask = ~with_pcr_mask
    
    data = selfcheck[with_pcr_mask] if with_pcr else selfcheck[without_pcr_mask]
    testset = np.random.choice(data['patient_id'].unique(), int(len(data['patient_id'].unique())*percentage), replace=False)
    chunk1_mask = ~data['patient_id'].isin(testset)
    
    chunk1 = data[chunk1_mask].copy()
    chunk2 = data[~chunk1_mask].copy()
    
    return chunk1, chunk2

@timer_decorator
def filter_dict_records(dict_records):
    return [{k: v for k, v in x.items() if not pd.isna(v)} for x in dict_records]

@timer_decorator
def get_baseline(group):
    if group is None:
        return None
    filtered_data = filter_dict_records(group.to_dict('records'))
    dataframes = [create_dataframe(data) for data in filtered_data]
    return pd.concat(dataframes) if dataframes else None

@timer_decorator
def create_dataframe(data):
    if 'days_from_init' not in data:
        return pd.DataFrame(columns=['r_days', 'var_name', 'value'])
    return pd.DataFrame({
        'r_days': [data['days_from_init']] * len(data),
        'var_name': data.keys(),
        'value': data.values()
    })

# %%
@timer_decorator
def pcr_date_check(pcr_df, group, pat_id, two_weeks=pd.Timedelta(14, 'd')):
    pcr_dates = pcr_df.loc[pcr_df['patient_id']==pat_id, 'pcr_date']
    pcr_dates = pcr_dates[pd.notna(pcr_dates)]
    pcr_results = pcr_df.loc[(pcr_df['patient_id'] == pat_id) & pcr_df['pcr_date'].isin(pcr_dates), 'pcr_result'].values - 1
    
    def _filter_group(group_df, pcr_date):
        pcr_date = pd.Timestamp(pcr_date)
        group_filtered = group_df.loc[group_df['selfcheck_date'].between(pcr_date - two_weeks, pcr_date + two_weeks)]
        if not group_filtered.empty:
            group_filtered = group_filtered.assign(days_from_init=(group_filtered['selfcheck_date'] - group_filtered['selfcheck_date'].min()).dt.days)
            return group_filtered.sort_values('selfcheck_date').reset_index(drop=True)
        else:
            return None
    
    results = [(pcr_result, _filter_group(group.groupby('patient_id').get_group(pat_id), pcr_date)) 
               for pcr_date, pcr_result in zip(pcr_dates, pcr_results)]
    
    return results


# %% 
@timer_decorator
def get_target_dates(df, date_column = 'selfcheck_date'):
    return [str(x.date()) for x in df[date_column].tolist()]

@timer_decorator
def get_demographic_df(patient_id, merged):
    if not merged['patient_id'].isin([patient_id]).any():
        return pd.DataFrame(columns=['r_days', 'var_name', 'value'])
    return pd.DataFrame([
        [0, 'birthday', merged.loc[merged['patient_id'] == patient_id, 'birthday'].iloc[0]],
        [0, 'gender', merged.loc[merged['patient_id'] == patient_id, 'gender'].iloc[0]],
    ], columns=['r_days', 'var_name', 'value'])

@timer_decorator
def get_geoloc_info_df(pat_id, target_dates):
    index_values = [(f"{pat_id}__{date}") for date in target_dates]
    pat_df = geo_df[geo_df.index.isin(index_values)].dropna(how='all', axis=1)
    
    if pat_df.empty:
        return pd.DataFrame(columns=['r_days', 'var_name', 'value'])
    
    temp_df = (
        pat_df
        .unstack()
        .reset_index(name='value')
        .rename(columns={'level_0': 'var_name'})
        .assign(
            r_days_ori=lambda x: pd.to_datetime(x['index'].str.split('__').str[1]).dt.date,
            r_days=lambda x: (x['r_days_ori'] - x['r_days_ori'].min()).dt.days.astype(int),
        )
        [['r_days', 'var_name', 'value']]
        .reset_index(drop=True)
        .pipe(lambda df: df[~df['value'].isin(['NaN', 'nan', np.nan])].reset_index(drop=True))
    )

    return temp_df


@timer_decorator
def get_metaCoV_df(pat_id, target_dates):
    temp_list = []
    pat_df = ref_kr[ref_kr['date'].isin(target_dates)]
    pat_df['date'] = (pd.to_datetime(pat_df['date']) - pd.to_datetime(pat_df['date'].min())).dt.days
    cols = pat_df.columns[1:]
    if not pat_df.empty:
        temp_list += [[row.date, f'{col}', getattr(row, col)]
                      for row in pat_df.itertuples() for col in cols
                      if getattr(row, col) is not None and str(getattr(row, col)).lower() not in ['nan', 'NaN']]


    return pd.DataFrame(temp_list, columns=['r_days', 'var_name', 'value'])

# %% 
@timer_decorator
def finalizer(base_df, demo_infos, geoloc_infos, metacov_infos):
    # Concatenate the four dataframes into one
    merged_df = pd.concat([df for df in [base_df, demo_infos, geoloc_infos, metacov_infos] if df is not None])
    no_cols = ['selfcheck_date', 'days_from_init', 'hx_covid_date', 'hx_covid_result', 'epi_num','patient_id']
    merged_df = merged_df[~merged_df['var_name'].isin(no_cols)]
    merged_df = merged_df.sort_values(['r_days', 'var_name']).drop_duplicates().reset_index(drop=True)

    # Convert values to float
    merged_df['value'] = pd.to_numeric(merged_df['value'], 'ignore').dropna()

    # Check if there are any non-convertible values
    non_float_indices = merged_df['value'].isna()

    if not non_float_indices.any():
        merged_df['r_days'] = merged_df['r_days'].apply(lambda x: f'{x * 24:02.0f}:00:00')
        return merged_df

    print(f"Values at indices {non_float_indices[non_float_indices].index.tolist()} in '{pat_id}' are not convertible to floats.")
    print(merged_df.iloc[non_float_indices[non_float_indices].index.tolist()])
    return None

# %% 
@timer_decorator
def save_csv_and_collect_labels(my_df, label, pat_id, label_collector, input_col, epi_num=None):
    folder = 'pos' if label == 1 else 'neg'
    if len(my_df['r_days'].unique()) > 1:
        filename = f'{pat_id}_{epi_num}' if epi_num else pat_id
        my_df.to_csv(f'./{folder}/{filename}.csv', encoding='utf-8-sig', index=False)
        label_collector.append([filename, label])
        input_col.append(my_df['var_name'].unique().tolist())

# %% 
#@timer_decorator
def assign_episodes_v2(group):
    chunks = np.array_split(group, np.ceil(len(group) / 60))
    dfs = []
    for i, chunk in enumerate(chunks):
        chunk = chunk.sort_values('selfcheck_date').reset_index(drop=True)
        time_gap = chunk['selfcheck_date'] - chunk['selfcheck_date'].shift()
        sep = np.where(time_gap > pd.Timedelta('168h'))[0]
        sep = np.concatenate(([0], sep, [len(chunk)]))
        epi_num = np.arange(len(sep) - 1) + 1 + i*100
        chunk['epi_num'] = np.repeat(epi_num, np.diff(sep))
        dfs.append(chunk)
    group = pd.concat(dfs)

    return group


def drop_single_epi(df, column_to_filter_by='epi_num'):
    # Get the unique values that have only one row associated with them
    unique_values = df[column_to_filter_by].value_counts()[lambda x: x == 1].index.tolist()
        
    # Filter out all rows in the dataset that have any of the unique values
    df_filtered = df[~df[column_to_filter_by].isin(unique_values)]
    
    return df_filtered


def filter_dates(dates):
    dates = [datetime.strptime(date, '%Y-%m-%d') for date in sorted(dates)]
    filtered_dates = [dates[0]] + [dates[i] for i in range(1, len(dates))
                                    if (dates[i] - dates[i-1]) > timedelta(days=60)]
    return filtered_dates 

def wo_pcr_prepro(wo_pcr_train):
    temp_list = []
    for k, g in tqdm(wo_pcr_train.groupby('patient_id')):

        covid_dates = [x for x in g['hx_covid_date'].unique() if str(x).lower()!='nan']

        if covid_dates:
            for cov_idx, cov_date in enumerate(filter_dates(covid_dates)):
                gi = g[(g['selfcheck_date'].between(pd.Timestamp(cov_date) - pd.Timedelta(days=14), pd.Timestamp(cov_date) + pd.Timedelta(days=14)))]
                gi = gi.assign(epi_num=cov_idx)
                temp_list.append(gi)

        if len(g) > 1:
            #g = g.assign(label=0)
            temp_list.append(drop_single_epi(assign_episodes_v2(g)))

    epiNum_assigned = pd.concat(temp_list)
    return epiNum_assigned

@timer_decorator
def nonpcr_date_check(group):
    group['days_from_init'] = (group['selfcheck_date'] - group['selfcheck_date'].min()).dt.days
    label = 1 if (group['hx_covid_result'] == 1).any() else 0
    return label, group

# %% 
@timer_decorator
def split_data(pos_dir, neg_dir, train_dir, test_dir, proportion=0.8):
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    files = {}
    for directory, label in [(pos_dir, 'pos'), (neg_dir, 'neg')]:
        for filename in os.listdir(directory):
            file_id = filename.split("_")[0]
            files.setdefault(file_id, {'pos': [], 'neg': []})
            files[file_id][label].append(os.path.join(directory, filename))

    ids = list(files.keys())
    random.shuffle(ids)
    split_index = int(len(ids) * proportion)

    for file_id in ids:
        dest_dir = train_dir if ids.index(file_id) < split_index else test_dir
        src_files = list(itertools.chain.from_iterable(files[file_id].values()))
        for src_file in src_files:
            dest_file = os.path.join(dest_dir, os.path.basename(src_file))
            if not os.path.exists(dest_file):
                shutil.copy(src_file, dest_file)

    for directory in [pos_dir, neg_dir, train_dir, test_dir]:
        num_files = len([filename for filename in os.listdir(directory)])
        print(f"Number of files in {directory}: {num_files}")


@timer_decorator
def save_labels_and_inputdict(prefix):
    labels = pd.DataFrame(label_collector, columns=['filename', 'label'])
    labels.to_csv(f'{prefix}_labels.csv', index=False)
    print(f"pos={labels['label'].sum()}, total={len(labels)}", os.linesep)

    inputdict = sorted(set(list(itertools.chain(*input_col))))
    inputdict = {v:k for k,v in enumerate(inputdict)}
    print(inputdict)

    with open(f'{prefix}_inputdict.p', 'wb') as h:
        pickle.dump(inputdict, h)
        
@timer_decorator
def compress_csv_files(directory_path, archive_name, remove=False):
    csv_files = [f for f in os.listdir(directory_path) if f.endswith('.csv')]
    with tarfile.open(archive_name, 'w:gz') as tar:
        tar.add(directory_path, arcname=os.path.basename(directory_path))
    if remove:
        for csv_file in csv_files:
            os.remove(os.path.join(directory_path, csv_file))
        os.rmdir(directory_path)
        print('CSV files and directory have been removed')

# %% 
@timer_decorator
def prepare_geo_df(df):
    df = df.set_index('index')
    return df[['path_sum', 'loc_std']]

# %% 
def with_pcr_process(group):
    # Get patient IDs and PCR dates
    pat_id = group.name

    # Get list of group-label pairs
    group_label_pairs = pcr_date_check(pcr_df, group, pat_id)
    
    # Iterate through group-label pairs
    for epi_num, (label, group_filtered) in enumerate(group_label_pairs):
        baseline = get_baseline(group_filtered)
        if baseline is None:
            continue
        target_dates = get_target_dates(group_filtered)

        # Get demographic, geolocation, and metaCoV information dataframes
        demo_infos = get_demographic_df(pat_id, merged_others)
        geoloc_infos = None
        #geoloc_infos = get_geoloc_info_df(pat_id, target_dates)
        metacov_infos = get_metaCoV_df(pat_id, target_dates)
        #metacov_infos = None

        # Concatenate all dataframes
        finalized_df = finalizer(baseline, demo_infos, geoloc_infos, metacov_infos)
        if finalized_df is not None:
            # Save the processed data to CSV file
            save_csv_and_collect_labels(finalized_df, label, pat_id, label_collector, input_col, epi_num)

# %% 
# Define the paths to the CSV files
paths = ['path/to/dir']
# Call the read_csv_files() function to read all the CSV files and assign the dataframes to more descriptive variables
files = read_csv_files(paths)
user_df, underlingdz_df, gps, selfcheck_df, pcr_df = files

# Call the select_reference_data() function to load the reference CSV file and select the specific columns for Korea (KOR)
ref_kr = select_reference_data('../owid-covid-data.csv')

#temporary treatment
geo_df = prepare_geo_df(gps)

# %% 
no_cols = [selfcheck_df.columns[0],'_id','contact_relation', 'etc_status', 
           'reg_date','isolation_date', 'abroad_country', 
           'abroad_enter_date', 'visit_region', 'visit_date']

selfcheck_df_processed = (selfcheck_df.pipe(drop_n_to_dt, no_cols)
                                      .pipe(create_dummy_variables_v3, 'selfcheck_reason')
                                      .pipe(create_dummy_variables_v3, 'sx_name', prefix='')
                                      .pipe(remove_pregnant_rogues)
                                      .pipe(remove_bodytemp_rogues))

merged_others = processing_others(user_df, pcr_df, underlingdz_df)

# %% 
symptom_dict = {'가래': 'Phlegm',
 '가슴통증': 'Chest_pain',
 '근육통': 'Muscle_pain',
 '기타': 'Other_symptoms',
 '눈 충혈': 'Eye_congestion',
 '늘어지고\n피곤함': 'Fatigue_and_weakness',
 '두통': 'Headache',
 '마른 기침': 'Dry_cough',
 '몸통발진': 'Body_rash',
 '미각소실': 'Loss_of_taste',
 '변비': 'Constipation',
 '설사': 'Diarrhea',
 '숨참': 'Shortness_of_breath',
 '오한': 'Chills',
 '인후통(목통증)': 'Sore_throat_(neck_pain)',
 '입맛없음': 'Loss_of_appetite',
 '입술 주변 물집': 'Blister_around_the_lips',
 '입술 파래짐': 'Swelling_of_the_lips',
 '재채기': 'Sneezing',
 '콧물': 'Runny_nose',
 '피가 섞인 기침': 'Coughing_up_blood',
 '후각소실': 'Loss_of_smell'}

selfcheck_df_processed = selfcheck_df_processed.rename(columns=symptom_dict)

# %%
cols_to_fillna = ['is_abroad', 'is_visit', 'is_contact', 'is_isolation',\
                  'fever', 'fever_period', 'oxygen_therapy']
selfcheck_df_processed[cols_to_fillna] = selfcheck_df_processed[cols_to_fillna].fillna(0)


TIMER_ENABLED = False
PCR = True
create_folders()

if PCR:
    
    # Load data
    label_collector = []
    input_col = []
    
    # Use apply() to process each patient's data in parallel
    w_pcr_train, w_pcr_test = get_chunks(selfcheck_df_processed, pcr_df, percentage=0, with_pcr=True)
    w_pcr_train.groupby('patient_id').progress_apply(with_pcr_process)
        
    compress_csv_files("./pos", "with_pcr_pos.tgz")
    compress_csv_files("./neg", "with_pcr_neg.tgz") 
    
    split_data('pos', 'neg', 'train_fevercoach', 'test_fevercoach')
    save_labels_and_inputdict("with_pcr")
    
    compress_csv_files("./train_fevercoach", "with_pcr_train.tgz", remove=True)
    compress_csv_files("./test_fevercoach", "with_pcr_test.tgz", remove=True)

PCR = True
TIMER_ENABLED = False
create_folders()


if not PCR:

    def without_pcr_process(group):
        # Get label and filtered group data for each patient
        label, group = nonpcr_date_check(group)
        baseline = get_baseline(group)

        if baseline is None:
            return None

        taget_dates = get_target_dates(group)
        
        # Get demographic, geolocation, and metaCoV information dataframes
        demo_infos = get_demographic_df(group['patient_id'].iloc[0], merged_others)
        geoloc_infos = get_geoloc_info_df(group['patient_id'].iloc[0], taget_dates)
        metacov_infos = get_metaCoV_df(group['patient_id'].iloc[0], taget_dates)

        # Concatenate all dataframes
        finalized_df = finalizer(baseline, demo_infos, geoloc_infos, metacov_infos)
        if finalized_df is not None:
            # Save the processed data to CSV file
            save_csv_and_collect_labels(finalized_df, label, group['patient_id'].iloc[0], label_collector, input_col, group['epi_num'].iloc[0])

    # Apply the process_group function to each group and collect the results
   
    label_collector = []
    input_col = []
    
    wo_pcr_train, wo_pcr_test = get_chunks(selfcheck_df_processed, pcr_df, 0, with_pcr=False)
    epiNum_assigned = wo_pcr_prepro(wo_pcr_train)
    epiNum_assigned.groupby(['patient_id', 'epi_num']).progress_apply(without_pcr_process)    

    compress_csv_files("./pos", "without_pcr_pos.tgz")
    compress_csv_files("./neg", "without_pcr_neg.tgz") 
    
    split_data('pos', 'neg', 'train_fevercoach', 'test_fevercoach')
    save_labels_and_inputdict("without_pcr")
    
    compress_csv_files("./train_fevercoach", "without_pcr_train.tgz", remove=True)
    compress_csv_files("./test_fevercoach", "without_pcr_test.tgz", remove=True)
