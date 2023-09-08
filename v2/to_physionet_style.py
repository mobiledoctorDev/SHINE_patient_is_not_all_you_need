#%%
import pandas as pd
import warnings
import os
import datetime
import gc
from tqdm.auto import tqdm
import shutil, random
import pickle

warnings.simplefilter(action='ignore', category=FutureWarning)
os.umask(0)

# from pandarallel import pandarallel
# pandarallel.initialize(nb_workers=4, progress_bar=True)

#%%
def iter_chunk_by_id(file):
    csv_reader = pd.read_csv(file, iterator=True, chunksize=1)
    first_chunk = csv_reader.get_chunk()
    _id = first_chunk.iloc[0, 1]
    chunk = pd.DataFrame(first_chunk)
    for row in csv_reader:
        if _id == row.iloc[0, 1]:
            _id = row.iloc[0, 1]
            chunk = chunk.append(row)
            continue
        _id = row.iloc[0, 1]
        yield chunk
        chunk = pd.DataFrame(row)
    yield chunk


#%%
def epinum_assign(g):
    g = g.sort_values('date').reset_index(drop=True)
    time_gap = [pd.to_datetime(g['date'].tolist()[x + 1]) - pd.to_datetime(g['date'].tolist()[x]) for x in
                range(len(g['date'].tolist()) - 1)]
    sep = [k for k, v in enumerate(time_gap) if v > pd.Timedelta('72h')]
    sep = sorted(set(sep + [0, len(g.index)]))

    end = 0
    for epi_num, pos in enumerate(sep):
        if pos < len(g.index):
            # print(g.index[end:sep[epi_num+1]+1])
            g.at[g.index[end:sep[epi_num + 1] + 1], 'epi_num'] = epi_num + 1
            end = sep[epi_num + 1] + 1

    g['data_1'] = g['data_1'].astype(str)
    return g


# %%
def extract_and_save(epi_num_assigned):
    for k, g in (epi_num_assigned.groupby(['baby_id', 'epi_num'])):
        if not g.loc[(g['type'] == 4) & (g['data_1'] == '3')].empty:
            single_case = []

            if not g.loc[(g['type'] == 1)].empty:  # 체온
                for idx, _temp_dict in enumerate(g.loc[(g['type'] == 1)].to_dict('records')):
                    try:
                        if not str(_temp_dict['data_2']).lower() == 'f':
                            single_case.append([_temp_dict['date'], 'body_temperature', float(_temp_dict['data_1'])])
                    except:
                        print(_temp_dict)
                    # data_2 = c/f, #data_3 = nfc/hs/dot, #data_4 = 0ear,1ax,2oral, #data_5: covid_vac, #lat, #lng, #weight
                    # 화씨는 버리는걸로 / 귀or null? / data_5 무시

            _type_dict = {'0': 'antipyretic', '1': 'other_drugs', '2': 'antibiotics'}
            if not g.loc[(g['type'] == 2)].empty:  # 해열제
                for idx, _temp_dict in enumerate(g.loc[(g['type'] == 2)].to_dict('records')):
                    try:
                        single_case.append([_temp_dict['date'], _type_dict[str(int(float(_temp_dict['data_1'])))],
                                            float(_temp_dict['data_2'])])
                    except:
                        print(_temp_dict)
                    # data_4 1:acet, 2:ibu, 3:dexi, #data_6 covid drug #drug_formulation 1 liquid, 2powder, 3, #convulsion
                    ##알약인 애들만 모아서 용량 분포
                    ##성분 무시

            if not g.loc[(g['type'] == 101)].empty:  # covid
                for idx, _temp_dict in enumerate(g.loc[(g['type'] == 2)].to_dict('records')):
                    if not _temp_dict['data_1'] == 0:
                        single_case.append([_temp_dict['date'], 'covid_diag', float(_temp_dict['data_1'])])

            _type_dict = {'1': 'symptom', '2': 'antibiotics', '3': 'diagnosis', '4': 'daily_records',
                          '5': 'vaccination'}
            if not g.loc[(g['type'] == 4)].empty:  # 메모
                for idx, _temp_dict in enumerate(g.loc[(g['type'] == 4)].to_dict('records')):
                    try:
                        if not _type_dict[str(int(float(_temp_dict['data_1'])))] == 'symptom':
                            single_case.append([_temp_dict['date'], _type_dict[str(int(float(_temp_dict['data_1'])))],
                                                float(_temp_dict['data_2'])])
                        else:
                            for vals in str(_temp_dict['data_2']).split('_'):
                                single_case.append(
                                    [_temp_dict['date'], f"{_type_dict[str(int(float(_temp_dict['data_1'])))]}_{vals}", 1])

                    except:
                        print(_temp_dict)


            if '8' in g.loc[(g['type'] == 4) & (g['data_1'] == '3'), 'data_2'].values:
                label = 'flu'
            else:
                label = 'notflu'

            _type_dict = {'flu': 1, 'notflu': 0}
            physionet_style_df = pd.DataFrame(single_case, columns=['time', 'var_name', 'value']).sort_values('time')

            weather_ref = pd.read_csv('./OBS_ASOS_DD_20211209143339.csv')

            try:
                weather_date = pd.to_datetime(min(g['date'])).date()
                age_val = (pd.to_datetime(min(physionet_style_df['time'])) - pd.to_datetime(_temp_dict['birthday'])).days
                onetime_vars = pd.DataFrame([[max(physionet_style_df['time']), 'gender', _temp_dict['gender']],
                                             [max(physionet_style_df['time']), 'age', age_val],
                                             [max(physionet_style_df['time']), 'weight', _temp_dict['weight']],
                                             [max(physionet_style_df['time']), 'convulsion', _temp_dict['convulsion']],
                                             [max(physionet_style_df['time']), 'is_flu', _type_dict[label]]],
                                            columns=['time', 'var_name', 'value'])
                physionet_style_df = physionet_style_df.append(onetime_vars)
                physionet_style_df['time'] = physionet_style_df['time'] - min(physionet_style_df['time'])
                physionet_style_df = physionet_style_df.loc[~physionet_style_df['var_name'].isin(['is_flu', 'diagnosis'])]
                physionet_style_df['time'] = (pd.to_datetime(physionet_style_df['time']) - pd.to_datetime(min(physionet_style_df['time'])))
                physionet_style_df = physionet_style_df.sort_values(by=['time'])
                physionet_style_df['time'] = physionet_style_df['time'].apply(lambda x: convert(x.total_seconds()))
                physionet_style_df = physionet_style_df.dropna().reset_index(drop=True)

                inputdict_col_append(df['var_name'].unique().tolist())
                label_col_append([os.path.splitext(file)[0], _type_dict[label]])
                date_col_append(weather_date)

                print(weather_ref.loc[weather_ref['date'] == weather_date])
                physionet_style_df.to_csv(f"{root_path}/{label}/{k[0]}_{int(k[1])}.csv", index=False)
            except:
                pass
# %%
def convert(seconds):
    #seconds = seconds % (24 * 3600)
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60

    return "%02d:%02d:%02d" % (hour, minutes, seconds)
#%%
root_path = f"./v2_episode/{str(datetime.datetime.now()).split(' ')[0]}"

if not (os.path.exists(f'{root_path}/flu')):
    os.makedirs(f'{root_path}/flu')

if not (os.path.exists(f'{root_path}/notflu')):
    os.makedirs(f'{root_path}/notflu')
# %%
#chunk_iter = iter_chunk_by_id("/opt/project/md_data__with_baby_having_dx_sorted.csv")
df =  pd.read_csv("/opt/project/md_data__with_baby_having_dx_sorted.csv", \
                  usecols=['baby_id', 'date', 'type', 'data_1', 'data_2', 'gender', 'birthday', 'weight', 'convulsion' ])

weather_ref = pd.read_csv('./OBS_ASOS_DD_20211209143339.csv')

date_collector = []
inputdict_collector = []
label_collector = []

date_col_append = date_collector.append
label_col_append = label_collector.append
inputdict_col_append = inputdict_collector.append

for no, chunk in tqdm(df.groupby('baby_id')):
    if not chunk.loc[(chunk['type']==4) & (chunk['data_1']=='3')].empty:
        extract_and_save(epinum_assign(chunk))
        gc.collect()

pd.DataFrame(labels).to_csv('/opt/project/v2_episode/labels.csv', index=False, encoding='utf-8-sig')
with open('/opt/project/v2_episode/inputdict.p', 'wb') as handle:
    pickle.dump(set(list(itertools.chain(*temp_list))), handle, protocol=pickle.HIGHEST_PROTOCOL)