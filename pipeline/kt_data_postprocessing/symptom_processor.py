import pandas as pd
from common import apply_and_concat 

class SymptomProcessor:
    def __init__(self):
        self.sx_name_dicts = {'마른 기침': '104',
        '피가 섞인 기침': '108',
        '가래': '107',
        '가슴통증': '102',
        '콧물': '106',
        '인후통(목통증)': '105',
        '숨참': '101',
        '근육통': '303',
        '두통': '302',
        '오한': '301',
        '미각소실': '309',
        '후각소실': '310',
        '늘어지고\n피곤함': '311',
        '입맛없음': '201',
        '설사': '204',
        '변비': '206',
        '눈 충혈': '312',
        '입술 파래짐': '103',
        '재채기': '109',
        '몸통발진': '313',
        '입술 주변 물집': '314',
        '기타': '999'}

        self.symptom_columns = ['cough', 'sore_throat', 'shortness_of_breath',\
        'head_ache', 'runny_nose', 'muscle_pain', 'chills',\
        'loss_of_taste', 'loss_of_smell', 'sputum', 'chest_pain',\
        'indication_other', 'indication_abroad', 'indication_contact']
    
    def apply_and_concat(self, df, sx_name, is_contact, selfcheck_reason, func, cols_name):
        return pd.concat((df, df.apply(lambda x: pd.Series(func(x[sx_name], \
                                    x[is_contact], x[selfcheck_reason]), index=cols_name), axis=1)), axis=1)

    def symp_filter(self, sx_name, is_contact, selfcheck_reason):
        symptoms = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        purposes = [1, 0, 0] # 'indication_other', 'indication_abroad', 'indication_contact'
        if sx_name != None and sx_name != '':
            symptom_check_list = [self.sx_name_dicts[i] for i in sx_name.split('`')]
            if '104' or '108' in symptom_check_list: symptoms[0] = 1
            if '105' in symptom_check_list: symptoms[1] = 1 # 가래, 인후통 따로
            if '101' in symptom_check_list: symptoms[2] = 1 # 가슴통증따로
            if '302' in symptom_check_list: symptoms[3] = 1
            #### 여기부터 기존에 없는 증상
            if '106' in symptom_check_list: symptoms[4] = 1
            if '303' in symptom_check_list: symptoms[5] = 1
            if '301' in symptom_check_list: symptoms[6] = 1
            if '309' in symptom_check_list: symptoms[7] = 1
            if '310' in symptom_check_list: symptoms[8] = 1
            if '107' in symptom_check_list: symptoms[9] = 1
            if '102' in symptom_check_list: symptoms[10] = 1
        
        if is_contact == 1: purposes = [0, 0, 1]
        if selfcheck_reason == '6' and purposes[0] == 1: purposes = [0, 1, 0]
        return symptoms + purposes
    
    def make_symptom_columns(self, df):
        return self.apply_and_concat(df, 'sx_name', 'is_contact', \
                         'selfcheck_reason', self.symp_filter, self.symptom_columns)