# %%
import os
os.environ["OPENAI_API_KEY"] = ""
from urllib.parse import quote_plus
from sqlalchemy import create_engine
from llama_index.core.indices.struct_store.sql_query import (
    SQLTableRetrieverQueryEngine,
)
from llama_index.core.objects import (
    SQLTableNodeMapping,
    ObjectIndex,
    SQLTableSchema,
)
from llama_index.core import VectorStoreIndex, SQLDatabase
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.retrievers import NLSQLRetriever
from llama_index.llms.openai import OpenAI
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response.notebook_utils import display_source_node
from IPython.display import Markdown, display
from griptape.drivers.sql.sql_driver import SqlDriver


class Engine:
    def __init__(
        self,
        db_user,
        db_password,
        db_host,
        db_port,
        db_name,
        table_names,
        llm_model="gpt-4-turbo-preview",
        llm_temperature=0.1,
        embedding_model="text-embedding-3-large",
        similarity_top_k=3
    ):
        """
        Initialize the Engine with database and model configurations.
        
        Args:
            db_user (str): Database username
            db_password (str): Database password
            db_host (str): Database host
            db_port (str): Database port
            db_name (str): Database name
            table_names (list): List of table names to include
            llm_model (str): OpenAI LLM model to use
            llm_temperature (float): Temperature for LLM
            embedding_model (str): OpenAI embedding model to use
            similarity_top_k (int): Number of similar results to retrieve
        """
        self.db_user = db_user
        self.db_password = db_password
        self.db_host = db_host
        self.db_port = db_port
        self.db_name = db_name
        self.table_names = table_names
        self.llm_model = llm_model
        self.llm_temperature = llm_temperature
        self.embedding_model = embedding_model
        self.similarity_top_k = similarity_top_k
        
        # Initialize components
        self._setup_database_connection()
        self._setup_llm()
        self._setup_sql_database()
        self._setup_retrievers()
        self._setup_query_engines()
    
    def _setup_database_connection(self):
        """Set up the database connection."""
        encoded_password = quote_plus(self.db_password)
        self.connection_string = f"postgresql://{self.db_user}:{encoded_password}@{self.db_host}:{self.db_port}/{self.db_name}"
        self.engine = create_engine(self.connection_string)
    
    def _setup_llm(self):
        """Set up the LLM."""
        self.llm = OpenAI(temperature=self.llm_temperature, model=self.llm_model)
    
    def _setup_sql_database(self):
        """Set up the SQL database wrapper."""
        self.sql_database = SQLDatabase(self.engine, include_tables=self.table_names)
    
    def _setup_retrievers(self):
        """Set up the retrievers."""
        self.nl_sql_retriever = NLSQLRetriever(
            self.sql_database, 
            tables=self.table_names, 
            llm=self.llm, 
            return_raw=True
        )

    def get_table_schema(self, table_name):
        """Get the schema for a specific table."""
        driver = SqlDriver(engine_url=self.connection_string)
        return driver.get_table_schema(table_name)
    
    def _setup_query_engines(self):
        """Set up the query engines."""
        # Basic retriever query engine
        self.basic_query_engine = RetrieverQueryEngine.from_args(
            self.nl_sql_retriever, 
            llm=self.llm
        )
        
        # Advanced table retriever query engine with embeddings
        table_node_mapping = SQLTableNodeMapping(self.sql_database)
        table_schema_objs = []
        
        # Get table schemas for all tables
        for table_name in self.table_names:
            table_schema = self.get_table_schema(table_name)
            table_schema_objs.append(
                SQLTableSchema(table_name=table_name, context_str=table_schema)
            )
        
        obj_index = ObjectIndex.from_objects(
            table_schema_objs,
            table_node_mapping,
            VectorStoreIndex,
            embed_model=OpenAIEmbedding(model=self.embedding_model),
        )
        
        self.advanced_query_engine = SQLTableRetrieverQueryEngine(
            self.sql_database, 
            obj_index.as_retriever(similarity_top_k=self.similarity_top_k)
        )
    
    
    
    def retrieve(self, query):
        """Retrieve results using the NL SQL retriever."""
        return self.nl_sql_retriever.retrieve(query)
    
    def query_basic(self, query):
        """Query using the basic query engine."""
        return self.basic_query_engine.query(query)
    
    def query_advanced(self, query):
        """Query using the advanced query engine with embeddings."""
        return self.advanced_query_engine.query(query)
    
    def display_results(self, results):
        """Display retrieval results."""
        for node in results:
            display_source_node(node)
    
    def display_response(self, response):
        """Display query response in markdown format."""
        display(Markdown(f"<b>{response}</b>"))


# Example usage:
# engine = Engine(
#     db_user="postgres.winxhexdlkffhwxbxwve",
#     db_password="2030D@ta",
#     db_host="aws-0-us-east-2.pooler.supabase.com",
#     db_port="5432",
#     db_name="postgres",
#     table_names=["test_table"],
#     llm_model="gpt-4-turbo-preview",
#     llm_temperature=0.1
# )
#
# # Use the engine
# results = engine.retrieve("KICで発生したトラブルの件数を教えてください。")
# response = engine.query_basic("2021年4月19日に発生したNFCUDL927-1500におけるトラブルの原因と対策について説明してください。")
# advanced_response = engine.query_advanced("21年度四日市工場において最も損失金額が大きいトラブル内容と損失金額を教えてください。")



