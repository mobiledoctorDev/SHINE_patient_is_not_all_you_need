from configparser import ConfigParser
import pandas as pd
import sqlalchemy
from sqlalchemy.pool import NullPool

class DBConnector:
    def __init__(self, path_to_yml='../../dh_scripts/secret.ini'):
        config = ConfigParser()
        config.read(path_to_yml)

        self.path_to_yml = path_to_yml
        self.hostname = config["kt_db"]["HOSTNAME"]
        self.port = int(config["kt_db"]["PORT"])
        self.username = config["kt_db"]["USERNAME"]
        self.password = config["kt_db"]["PASSWORD"]
        self.database = config["kt_db"]["DATABASE"]
        url = f"postgresql://{self.username}:{self.password}@{self.hostname}:{self.port}/{self.database}"
        self.kt_engine = sqlalchemy.create_engine(url, client_encoding='utf8', poolclass=NullPool)
    
    def get_df(self, qry):
        return pd.read_sql(qry, self.kt_engine)