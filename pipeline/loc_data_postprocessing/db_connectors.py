import os
from datetime import datetime, timedelta
import logging
import sqlalchemy
from sqlalchemy.pool import NullPool
import psycopg2
import mysql
import mysql.connector
import yaml
import pandas as pd

logger = logging.getLogger(__name__)

KT_SHINE_DB_YML_PATH = f"{os.path.dirname(__file__)}/configs/kt_db_conf.yml"


class DBCredential:
    def __init__(self, host, port, dbname, user, password):
        self.host = host
        self.port = port
        self.dbname = dbname
        self.user = user
        self.password = password

    @classmethod
    def from_yml(cls, path_to_yml):
        """ Factory function of DBCredential. Child classes can use this function."""
        with open(path_to_yml) as f:
            db_conf = yaml.load(f, Loader=yaml.FullLoader)

        host = db_conf["host"]
        dbname = db_conf['dbname']
        user = db_conf['user']
        password = db_conf['password']
        port = db_conf['port']
        return cls(host=host, port=port, dbname=dbname, user=user, password=password)


class PostgresConnector(DBCredential):
    @property
    def url(self):
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.dbname}"

    def get_engine(self):
        return sqlalchemy.create_engine(self.url, client_encoding='utf8', poolclass=NullPool)


class BaseDBConnector:
    """Abstract class for the DB connection. """
    QUERY_TEMPLATE = "SELECT {} + {x} + {y};"

    def __init__(self, db_config_filepath, cls_db_type):
        self.credential = cls_db_type.from_yml(db_config_filepath)

    def get_engine(self):
        return self.credential.get_engine()

    def run(self, *args, **kwargs):
        query = self.QUERY_TEMPLATE.format(*args, **kwargs)
        return self.query_run(query)

    def query_run(self, query):
        engine = self.get_engine()
        with engine.connect() as conn:
            ret = conn.execute(query)
            rows = ret.fetchall()
            return rows

    @staticmethod
    def get_basedbconnector(conf):
        if conf['dbname'] == 'research':
            return BaseDBConnector(KT_SHINE_DB_YML_PATH, PostgresConnector)
        else:
            raise ValueError("Wrong DB name")


class LowLevelConnector:
    """You should explict the column names of the target SQL. This class doesn't know the columns"""
    @staticmethod
    def _query_run(query, conf):
        conn = mysql.connector.connect(
            host=conf['host'],
            port=conf['port'],
            database=conf['dbname'],
            user=conf['user'],
            password=conf['password'])

        print("conn:", conn)
        cursor = conn.cursor()
        cursor.execute(query)
        rows = []
        if query[:6] == 'SELECT' or query[:6] == 'select':
            rows = cursor.fetchall()
        try:
            conn.commit()
        except:
            conn.rollback()
        conn.close()
        return rows

    @staticmethod
    def create_df(query, cols, conf):
        rows = LowLevelConnector._query_run(query, conf)
        return pd.DataFrame(rows, columns=cols, dtype=str)

    @staticmethod
    def get_conf(FILE_PATH):
        with open(FILE_PATH) as f:
            return yaml.load(f, Loader=yaml.FullLoader)
