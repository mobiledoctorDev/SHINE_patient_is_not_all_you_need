import pandas as pd
from common import apply_and_concat 

class LocationProcessor:
    def __init__(self):
        df_sigungu_name = pd.read_csv('./data/sigungu_name.csv')
        sigungu_dict = {}
        for addr_1, addr_2 in df_sigungu_name.values:
            if addr_1 not in sigungu_dict:
                sigungu_dict[addr_1] = []
            sigungu_dict[addr_1].append(addr_2)
        self.sigungu_dict = sigungu_dict

        self.df_sigungu_population_raw = pd.read_csv('./sigungu_population/population_1.csv', encoding='cp949')
        for i in range(2, 18):
            df_append = pd.read_csv(f'./sigungu_population/population_{i}.csv', encoding='cp949')
            self.df_sigungu_population_raw = pd.concat((self.df_sigungu_population_raw, df_append))
        self.df_sigungu_population_raw.reset_index(drop=True)
        self.df_sigungu_population_raw.head()

    def make_sidosigungu(self, location):
        sido = ''
        sigungu = ''
        if type(location) == float:
            return sido, sigungu
        for addr1 in self.sigungu_dict:
            if addr1 in location:
                sido = addr1
                for addr2 in self.sigungu_dict[addr1]:
                    if addr2 in location:
                        sigungu = addr2
        return sido, sigungu
    
    def limit_to_korea_location(self, df):
        df = df[df['lat'].notnull()]
        df = df[df['lng'].notnull()]
        df['lat'] = df['lat'].astype(float)
        df['lng'] = df['lng'].astype(float)
        df = df[(df['lat'] >= 33) & 
                    (df['lat'] <= 39) &
                    (df['lng'] >= 124) &
                    (df['lng'] <= 132)]

        return df
    
    def make_sidosigungu_columns(self, df):
        cols_name = ['sido', 'sigungu']
        df_loc = apply_and_concat(df, 'location', self.make_sidosigungu, cols_name)
        df_loc['sido'] = df_loc['sido'].fillna('')
        df_loc['sigungu'] = df_loc['sigungu'].fillna('')
        return df_loc
    
    def find_population(self, sido, sigungu):
        if sido == '충북':
            sido = '충청북도'
        elif sido == '충남':
            sido = '충청남도'
        elif sido == '경북':
            sido = '경상북도'
        elif sido == '경남':
            sido = '경상남도'
        elif sido == '전북':
            sido = '전라북도'
        elif sido == '전남':
            sido = '전라남도'
        condit1 = self.df_sigungu_population_raw['행정구역'].str.contains(sido)
        condit2 = self.df_sigungu_population_raw['행정구역'].str.contains(sigungu)
        df_res = self.df_sigungu_population_raw[condit1 & condit2]

        if df_res.shape[0] == 0:
            return None
        elif df_res.shape[0] == 1:
            return int(df_res['2022년06월_총인구수'].item().replace(',', ''))
        else:
            res = df_res['2022년06월_총인구수'].map(lambda x: int(x.replace(',', ''))).max()
            if sido == '인천' and sigungu == '동구':
                res = df_res[df_res['행정구역'] == '인천광역시 동구 (2814000000)']['2022년06월_총인구수'].item().replace(',', '')
            if sido == '경기' and sigungu == '양주':
                res = df_res[df_res['행정구역'] == '경기도 양주시 (4163000000)']['2022년06월_총인구수'].item().replace(',', '')
            if sido == '부산' and sigungu == '서구':
                res = df_res[df_res['행정구역'] == '부산광역시 서구 (2614000000)']['2022년06월_총인구수'].item().replace(',', '')
            return int(res)
    # def check_sido_sigungu(self, df):
