import os
from sqlalchemy import create_engine
from llama_index.core import SQLDatabase, ServiceContext
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from sqlalchemy import text
from llama_index.core.indices.struct_store.sql_query import NLSQLTableQueryEngine,SQLTableRetrieverQueryEngine
from llama_index.core.objects import (
    SQLTableNodeMapping,
    ObjectIndex,
    SQLTableSchema,
)
from llama_index.core import VectorStoreIndex
import urllib.parse
 

class Database(object):

    def __init__(self,db_name:str,user_name:str,pwd:str,host:str="localhost",port:str="5432"):

        self.db_user = user_name
        self.db_password = urllib.parse.quote_plus(pwd)
        self.db_host = host  
        self.db_name = db_name
        self.db_port = port
        self.query_engine = None

    @property
    def connection_uri(self,):
        connection_uri = f"postgresql+psycopg2://{self.db_user}:{self.db_password}@{self.db_host}/{self.db_name}"
        # connection_uri = f"postgresql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"
        return connection_uri
    
    def get_engine(self):
        try:
            engine = create_engine(self.connection_uri)
            with engine.connect() as connection_str:
                print('Successfully connected to the PostgreSQL database')
            return create_engine(self.connection_uri)
        except Exception as ex:
            print(f'Sorry failed to connect to db: {ex}')
            return None
        
    def get_sql_db(self,):
        sql_database = SQLDatabase(self.get_engine())
        return sql_database
    
    def build_query_engine(self,sql_database:SQLDatabase,tables:list[str]=None,sql_table_schemas:list=None):
        # we know the tables
        if isinstance(tables,list):
            self.query_engine = NLSQLTableQueryEngine(sql_database=sql_database,
                                                      tables=tables,
                                                      llm=Settings.llm)
        # we have the db table schema
        else:
            table_node_mapping = SQLTableNodeMapping(sql_database)
            obj_index = ObjectIndex.from_objects(sql_table_schemas,
                                                table_node_mapping,
                                                VectorStoreIndex)
            self.query_engine = SQLTableRetrieverQueryEngine(sql_database,
                                                              obj_index.as_retriever(similarity_top_k=1))
        print("query engine has been built.")
    
    def query_text(self,prompt:str):
        return self.query_engine.query(prompt)
    
    def query_sql(self,sql_cmd:str):
        engine = self.get_engine()
        with engine.connect() as con:
            out = con.execute(text(sql_cmd))
        return out