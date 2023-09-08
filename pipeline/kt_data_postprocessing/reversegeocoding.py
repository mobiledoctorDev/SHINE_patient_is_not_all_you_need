import pandas as pd
from sklearn.neighbors import KDTree

class ReverGeoCoder:
    def __init__(self, src_file='./data/location_korea_20210624.csv'):
        df_location = pd.read_csv(src_file)
        df_location = df_location[df_location['address_1'].notnull()]
        df_location = df_location[['_id','loc_1','loc_2','address_1','address_2','address_3','address_4','address_5']]
        df_location = df_location.rename(columns={'_id':'location_id'})
        df_location = df_location.reset_index(drop=True)
        df_location['loc_1'] = df_location['loc_1'].astype(float)
        df_location['loc_2'] = df_location['loc_2'].astype(float)

        print("Model Source loaded:", df_location.shape)
        self.kdt = KDTree(df_location[['loc_1', 'loc_2']], leaf_size=30, metric='euclidean')
        self.df_location = df_location.copy()


    def query_one(self, lati, longi):
        ret = self.kdt.query([[lati, longi]], k=1, return_distance=False)
        print("matching index:", ret)  # Print index
        return self.df_location.loc[[x[0] for x in ret], :]
    
    def query_list(self, list_location):
        """list should have [lat, lng] as elements. Ex) [[35.18692, 126.8927], [37.69954, 127.1940], [37.26776, 126.9961]]"""
        ret = self.kdt.query(list_location, k=1, return_distance=False)
        print("matching index:", ret)  # Print index
        return self.df_location.loc[[x[0] for x in ret], :]
    
    
    def query_df(self, df_query):
        """df should have columns ['lat', 'lng']"""
        ret = self.kdt.query(df_query[['lat', 'lng']].copy(), k=1, return_distance=False)
        df_ret = self.df_location.loc[[x[0] for x in ret], :]
        df_ret['loca'] = (df_ret.apply(lambda x: 
                                           self.make_location(x['address_1'], x['address_2'],
                                                         x['address_3'], x['address_4']), axis=1))
        df_query = df_query.reset_index(drop=True)
        df_ret = df_ret.reset_index(drop=True)
        df_query['location'] = df_ret['loca']
        return df_query
    
    def make_location(self, addr1, addr2, addr3, addr4):
        if type(addr1) != str: addr1 = ''
        if type(addr2) != str: addr2 = ''
        if type(addr3) != str: addr3 = ''
        if type(addr4) != str: addr4 = ''
            
        return f'{addr1} {addr2} {addr3} {addr4}'