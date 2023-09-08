import pandas as pd
from datetime import timedelta

class SurveillanceProcessor:
    def __init__(self):
        url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"
        self.df_timeseries_confirm = pd.read_csv(url)
    
    def filter_date(self, date):
        year = f"20{date[-2:]}"
        month = date[:date.index('/')]
        month = f'0{month}' if len(month) == 1 else month
        day = date[date.index('/') + 1 : date.rindex('/')]
        day = f'0{day}' if len(day) == 1 else day
        res = f'{year}-{month}-{day}'
        return res

    def get_global_confirmed_df(self):
        df_confirmed_global = self.df_timeseries_confirm.sum().reset_index()
        df_confirmed_global = df_confirmed_global.rename(columns={'index': 'global_confirmed_date', 0: 'global_confirmed_numsum'}).iloc[3:, :]
        df_confirmed_global = df_confirmed_global.reset_index(drop=True)
        df_confirmed_global['global_confirmed_num'] = df_confirmed_global['global_confirmed_numsum'] - df_confirmed_global.shift(1)['global_confirmed_numsum']
        df_confirmed_global = df_confirmed_global.dropna()
        df_confirmed_global['global_confirmed_num'] = df_confirmed_global['global_confirmed_num'].astype(int)
        df_confirmed_global['global_confirmed_date'] = df_confirmed_global['global_confirmed_date'].apply(self.filter_date)
        
        def global_find_week_confirmed_num(confirmed_date):
            date_column = 'global_confirmed_date'
            cur_datetime = pd.to_datetime(confirmed_date, format='%Y-%m-%d')
            before_week_datetime = cur_datetime - timedelta(days=7)
            before_week_date = before_week_datetime.strftime('%Y-%m-%d')
            df = df_confirmed_global[(before_week_date < df_confirmed_global[date_column]) & (df_confirmed_global[date_column] <= confirmed_date)]
            return int(df['global_confirmed_num'].mean() // 1)
        
        df_confirmed_global['global_confirmed_week_num'] = df_confirmed_global['global_confirmed_date'].apply(global_find_week_confirmed_num)
        return df_confirmed_global
    
    def get_kor_confirmed_df(self):
        rows = []
        for idx, values in enumerate(self.df_timeseries_confirm[self.df_timeseries_confirm['Country/Region'] == 'Korea, South'].iteritems()):
            if idx < 4: continue
            year = f"20{values[0][-2:]}"
            month = values[0][:values[0].index('/')]
            month = f'0{month}' if len(month) == 1 else month
            day = values[0][values[0].index('/') + 1 : values[0].rindex('/')]
            day = f'0{day}' if len(day) == 1 else day
            date = f'{year}-{month}-{day}'
            confirmed_num = values[1].item()
            rows.append([date, confirmed_num])
        df_numsum_confirmed = pd.DataFrame(rows, columns=['confirmed_date', 'confirmed_numsum'])
        df_numsum_confirmed['confirmed_num'] = df_numsum_confirmed['confirmed_numsum'] - df_numsum_confirmed.shift(1)['confirmed_numsum']
        df_numsum_confirmed = df_numsum_confirmed.dropna()
        df_numsum_confirmed['confirmed_num'] = df_numsum_confirmed['confirmed_num'].astype(int)
        
        def kor_find_week_confirmed_num(confirmed_date):
            date_column = 'confirmed_date'
            cur_datetime = pd.to_datetime(confirmed_date, format='%Y-%m-%d')
            before_week_datetime = cur_datetime - timedelta(days=7)
            before_week_date = before_week_datetime.strftime('%Y-%m-%d')
            df = df_numsum_confirmed[(before_week_date < df_numsum_confirmed[date_column]) & (df_numsum_confirmed[date_column] <= confirmed_date)]
            return int(df['confirmed_num'].mean() // 1)
        
        df_numsum_confirmed['confirmed_week_num'] = df_numsum_confirmed['confirmed_date'].apply(kor_find_week_confirmed_num)
        return df_numsum_confirmed
    
    def get_confirmed_df(self):
        global_confirmed_df = self.get_global_confirmed_df()
        kor_confirmed_df = self.get_kor_confirmed_df()
        return global_confirmed_df, kor_confirmed_df
    
    