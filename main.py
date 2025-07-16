"""
Database Application with Table Management and Text2SQL Integration
"""

import streamlit as st
import pandas as pd
import psycopg2
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.types import String, Integer, Float, DateTime, Boolean
import io
import sys
import os
from urllib.parse import quote_plus
import re
import json
from griptape.rules import Rule
from griptape.structures import Pipeline
from griptape.tasks import PromptTask

from text2sql_pipeline import Text2SQLPipeline

class DatabaseManager:
    """Manage database connections and operations."""
    
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.engine = None
        self.connect()
    
    def translate_column_names(self, japanese_columns: list) -> dict:
        """Translate Japanese column names to English using Griptape AI."""
        try:
            # Set OpenAI API key for Griptape from secrets
            import os
            try:
                os.environ["OPENAI_API_KEY"] = st.secrets["griptape"]["openai_api_key"]
            except KeyError:
                st.warning("Griptape OpenAI API key not found in secrets, using default configuration")
            
            # Create the prompt with all column names
            columns_text = ", ".join([f'"{col}"' for col in japanese_columns])
            
            pipeline = Pipeline()
            pipeline.add_tasks(
                PromptTask(
                    rules=[
                        Rule("Write your answer in json format"),
                        Rule("Translate Japanese column names to appropriate English database column names"),
                        Rule("Use snake_case format (lowercase with underscores)"),
                        Rule("Keep names descriptive but concise"),
                        Rule("Return a JSON object where keys are original Japanese names and values are English translations"),
                    ],
                ),
            )
            
            prompt = f"Translate these Japanese column names to English database column names: {columns_text}"
            result = pipeline.run(prompt)
            
            # Parse the JSON response
            try:
                # Extract JSON from the response
                response_text = str(result)
                # Find JSON content between braces
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group()
                    translation_dict = json.loads(json_str)
                    return translation_dict
                else:
                    st.warning("Could not parse translation response, using cleaned Japanese names")
                    return {}
            except json.JSONDecodeError:
                st.warning("Error parsing translation JSON, using cleaned Japanese names")
                return {}
                
        except Exception as e:
            st.warning(f"Translation service error: {str(e)}, using cleaned Japanese names")
            return {}
    
    def clean_column_name(self, name: str, is_japanese: bool = True) -> str:
        """Clean column name for database compatibility."""
        if pd.isna(name) or str(name).strip() == '':
            return f"column_{hash(str(name)) % 1000}"
        
        name = str(name).strip()
        
        if is_japanese:
            # For Japanese names, create a more readable romanized version
            # Replace common Japanese punctuation and spaces
            name = re.sub(r'[（）()【】\[\]「」『』・\s]+', '_', name)
            name = re.sub(r'[^\w\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF_]', '', name)
            # If still contains Japanese, keep it as is (PostgreSQL supports Unicode)
            if not name:
                return f"japanese_col_{hash(str(name)) % 1000}"
        else:
            # For English names, standard cleaning
            name = re.sub(r'[^\w\s]', '', name)
            name = re.sub(r'\s+', '_', name)
            name = name.lower()
        
        # Ensure it starts with a letter or underscore
        if not re.match(r'^[a-zA-Z_\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]', name):
            name = f"col_{name}"
        
        return name if name else f"unnamed_col_{hash(str(name)) % 1000}"
    
    def connect(self):
        """Establish database connection."""
        try:
            self.engine = create_engine(self.connection_string)
            # Test connection
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            st.success("✅ Database connected successfully!")
            return True
        except Exception as e:
            st.error(f"❌ Database connection failed: {str(e)}")
            return False
    
    def get_tables(self):
        """Get list of tables in the database."""
        try:
            inspector = inspect(self.engine)
            tables = inspector.get_table_names(schema='public')
            return tables
        except Exception as e:
            st.error(f"Error fetching tables: {str(e)}")
            return []
    
    def get_table_info(self, table_name: str):
        """Get table schema information."""
        try:
            inspector = inspect(self.engine)
            columns = inspector.get_columns(table_name, schema='public')
            return columns
        except Exception as e:
            st.error(f"Error fetching table info: {str(e)}")
            return []
    
    def get_table_data(self, table_name: str, limit: int = 100):
        """Get sample data from table."""
        try:
            query = f"SELECT * FROM {table_name} LIMIT {limit}"
            with self.engine.connect() as conn:
                df = pd.read_sql_query(query, conn)
            return df
        except Exception as e:
            st.error(f"Error fetching table data: {str(e)}")
            return pd.DataFrame()
    
    def upload_excel_to_table(self, excel_file, table_name: str, if_exists: str = 'replace', translate_columns: bool = False):
        """Upload Excel file to database table."""
        try:
            # Read Excel file
            df = pd.read_excel(excel_file)
            
            # Get original column names
            original_columns = df.columns.tolist()
            
            # Check if columns contain Japanese characters
            has_japanese = any(re.search(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]', str(col)) for col in original_columns)
            
            if translate_columns and has_japanese:
                st.info("🔄 Translating Japanese column names to English...")
                
                # Get translations using Griptape
                translation_dict = self.translate_column_names(original_columns)
                
                if translation_dict:
                    # Apply translations
                    new_columns = []
                    for col in original_columns:
                        if str(col) in translation_dict:
                            new_col = translation_dict[str(col)]
                            new_columns.append(self.clean_column_name(new_col, is_japanese=False))
                        else:
                            new_columns.append(self.clean_column_name(str(col), is_japanese=True))
                    
                    df.columns = new_columns
                    
                    # Show translation mapping
                    st.subheader("📝 Column Name Translations")
                    translation_df = pd.DataFrame({
                        'Original (Japanese)': original_columns,
                        'Translated (English)': new_columns
                    })
                    st.dataframe(translation_df, hide_index=True)
                else:
                    # Fallback to cleaned Japanese names
                    df.columns = [self.clean_column_name(str(col), is_japanese=True) for col in original_columns]
            else:
                # Clean column names while preserving Japanese characters
                cleaned_columns = []
                for col in original_columns:
                    cleaned_col = self.clean_column_name(str(col), is_japanese=has_japanese)
                    cleaned_columns.append(cleaned_col)
                
                df.columns = cleaned_columns
                
                if has_japanese:
                    st.info("📝 Using cleaned Japanese column names (PostgreSQL supports Unicode)")
            
            # Ensure no duplicate column names
            final_columns = []
            seen = set()
            for col in df.columns:
                if col in seen:
                    counter = 1
                    new_col = f"{col}_{counter}"
                    while new_col in seen:
                        counter += 1
                        new_col = f"{col}_{counter}"
                    final_columns.append(new_col)
                    seen.add(new_col)
                else:
                    final_columns.append(col)
                    seen.add(col)
            
            df.columns = final_columns
            
            # Upload to database
            df.to_sql(table_name, self.engine, if_exists=if_exists, index=False, schema='public')
            
            st.success(f"✅ Successfully uploaded {len(df)} rows to table '{table_name}'")
            return True
            
        except Exception as e:
            st.error(f"❌ Error uploading Excel file: {str(e)}")
            return False
    
    def execute_query(self, query: str):
        """Execute a SQL query and return results."""
        try:
            with self.engine.connect() as conn:
                df = pd.read_sql_query(query, conn)
            return df
        except Exception as e:
            st.error(f"Error executing query: {str(e)}")
            return pd.DataFrame()

class EnhancedText2SQLPipeline(Text2SQLPipeline):
    """Enhanced Text2SQL Pipeline with dynamic table selection."""
    
    def __init__(self, db_manager: DatabaseManager, selected_table: str = None):
        # Don't call super().__init__() to avoid the default database connection
        
        # Configuration from secrets
        try:
            api_key = st.secrets["openai"]["api_key"]
            azure_endpoint = st.secrets["openai"]["azure_endpoint"]
            api_version = st.secrets["openai"]["api_version"]
        except KeyError as e:
            st.error(f"❌ Missing OpenAI secret key: {e}")
            raise Exception(f"Missing OpenAI configuration in secrets: {e}")

        from llama_index.llms.azure_openai import AzureOpenAI
        from llama_index.core import Settings

        # Initialize LLM
        self.llm = AzureOpenAI(
            model="gpt-4o",
            deployment_name="gpt-4o",
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
        )

        # Set global settings
        Settings.llm = self.llm

        # Use the provided database manager
        self.db_manager = db_manager
        self.engine = db_manager.engine
        self.selected_table = selected_table
        
        # Get schema info for selected table
        if selected_table:
            self.schema_info = self._get_schema_info_for_table(selected_table)
        else:
            self.schema_info = self._get_all_schema_info()
    
    def _get_schema_info_for_table(self, table_name: str) -> str:
        """Get schema information for a specific table."""
        try:
            columns = self.db_manager.get_table_info(table_name)
            schema_text = f"Table: {table_name}\nColumns:\n"
            for col in columns:
                schema_text += f"- {col['name']} ({col['type']})\n"
            return schema_text
        except Exception as e:
            return f"Error getting schema for {table_name}: {str(e)}"
    
    def _get_all_schema_info(self) -> str:
        """Get schema information for all tables."""
        try:
            tables = self.db_manager.get_tables()
            schema_text = "Available Tables:\n"
            for table in tables:
                columns = self.db_manager.get_table_info(table)
                schema_text += f"\nTable: {table}\nColumns:\n"
                for col in columns:
                    schema_text += f"- {col['name']} ({col['type']})\n"
            return schema_text
        except Exception as e:
            return f"Error getting schema info: {str(e)}"
    
    def natural_language_to_sql(self, question: str) -> str:
        """Convert natural language question to SQL query."""
        
        table_context = ""
        if self.selected_table:
            table_context = f"Focus on the table: {self.selected_table}\n"
        
        prompt = f"""
        You are an expert SQL query generator. Given a natural language question, generate a PostgreSQL query.

        Database Schema:
        {self.schema_info}

        {table_context}
        Question: {question}

        Instructions:
        - Generate only the SQL query, no explanation
        - Use proper PostgreSQL syntax
        - Return only the SELECT statement
        - For Japanese text, use appropriate filtering
        - Use the public schema

        SQL Query:
        """
        
        response = self.llm.complete(prompt)
        sql_query = response.text.strip()
        
        # Clean up the SQL query (remove any markdown formatting)
        if sql_query.startswith("```sql"):
            sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
        elif sql_query.startswith("```"):
            sql_query = sql_query.replace("```", "").strip()
        
        return sql_query

def main():
    """Main Streamlit application."""
    
    st.set_page_config(
        page_title="Database Management & Text2SQL",
        page_icon="🗄️",
        layout="wide"
    )
    
    st.title("🗄️ Database Management & Text2SQL Pipeline")
    st.markdown("Connect to your database, manage tables, upload Excel files, and query with natural language!")
    
    # Initialize session state
    if 'db_manager' not in st.session_state:
        st.session_state.db_manager = None
    if 'selected_table' not in st.session_state:
        st.session_state.selected_table = None
    
    # Sidebar for database connection
    with st.sidebar:
        st.header("🔌 Database Connection")
        
        # Database connection info
        st.info("📡 Uses IPv4-compatible Supabase session pooler for better connectivity")
        st.info("🔐 Database credentials are loaded from Streamlit secrets")
        
        # Connect button (password comes from secrets)
        if st.button("Connect to Database"):
            try:
                # Get database password from secrets
                db_password = st.secrets["database"]["password"]
                
                # URL-encode the password to handle special characters
                encoded_password = quote_plus(db_password)
                # Use IPv4-compatible session pooler instead of direct connection
                connection_string = f"postgresql://postgres.winxhexdlkffhwxbxwve:{encoded_password}@aws-0-us-east-2.pooler.supabase.com:5432/postgres"
                
                st.session_state.db_manager = DatabaseManager(connection_string)
                
            except KeyError as e:
                st.error(f"❌ Missing secret key: {e}")
                st.error("Please configure database password in Streamlit secrets")
            except Exception as e:
                st.error(f"❌ Connection error: {e}")
        
        # Show connection status
        if st.session_state.db_manager:
            st.success("🟢 Connected")
        else:
            st.warning("🔴 Not Connected")
    
    # Main content
    if st.session_state.db_manager:
        
        # Create tabs
        tab1, tab2, tab3 = st.tabs(["📊 Table Management", "📤 Upload Excel", "🤖 Text2SQL Query"])
        
        with tab1:
            st.header("📊 Table Management")
            
            # Get and display tables
            tables = st.session_state.db_manager.get_tables()
            
            if tables:
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.subheader("Available Tables")
                    selected_table = st.selectbox("Select a table:", [""] + tables)
                    
                    if selected_table:
                        st.session_state.selected_table = selected_table
                        
                        # Show table info
                        st.subheader("Table Information")
                        table_info = st.session_state.db_manager.get_table_info(selected_table)
                        
                        info_data = []
                        for col in table_info:
                            info_data.append({
                                "Column": col['name'],
                                "Type": str(col['type']),
                                "Nullable": col.get('nullable', 'Unknown')
                            })
                        
                        if info_data:
                            df_info = pd.DataFrame(info_data)
                            st.dataframe(df_info, hide_index=True)
                
                with col2:
                    if selected_table:
                        st.subheader(f"Sample Data: {selected_table}")
                        
                        # Show sample data
                        sample_data = st.session_state.db_manager.get_table_data(selected_table, 20)
                        if not sample_data.empty:
                            st.dataframe(sample_data, hide_index=True)
                        else:
                            st.info("No data found in this table.")
            else:
                st.info("No tables found in the database.")
        
        with tab2:
            st.header("📤 Excel File Upload")
            
            uploaded_file = st.file_uploader("Choose an Excel file", type=['xlsx', 'xls'])
            
            if uploaded_file is not None:
                # Preview the Excel file
                try:
                    df_preview = pd.read_excel(uploaded_file)
                    st.subheader("Preview of Excel File")
                    st.dataframe(df_preview.head(), hide_index=True)
                    
                    # Table name and options
                    col1, col2 = st.columns(2)
                    with col1:
                        table_name = st.text_input("Table Name", value="new_table")
                    with col2:
                        if_exists_option = st.selectbox("If table exists:", ["replace", "append", "fail"])
                    
                    # Check if Excel has Japanese column names
                    has_japanese = any(re.search(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]', str(col)) for col in df_preview.columns)
                    
                    # Japanese column translation option
                    if has_japanese:
                        st.subheader("🇯🇵 Japanese Column Names Detected")
                        translate_columns = st.checkbox(
                            "🔄 Translate Japanese column names to English using AI",
                            help="Uses Griptape AI to translate Japanese column names to English snake_case format"
                        )
                        
                        if not translate_columns:
                            st.info("📝 Japanese column names will be cleaned but preserved (PostgreSQL supports Unicode)")
                    else:
                        translate_columns = False
                    
                    # Upload button
                    if st.button("Upload to Database"):
                        success = st.session_state.db_manager.upload_excel_to_table(
                            uploaded_file, table_name, if_exists_option, translate_columns
                        )
                        if success:
                            st.balloons()
                
                except Exception as e:
                    st.error(f"Error reading Excel file: {str(e)}")
        
        with tab3:
            st.header("🤖 Text2SQL Query")
            
            # Table selection for queries
            tables = st.session_state.db_manager.get_tables()
            query_table = st.selectbox(
                "Select table for queries (optional - leave empty for all tables):", 
                [""] + tables,
                key="query_table"
            )
            
            # Initialize Text2SQL pipeline
            if 'text2sql_pipeline' not in st.session_state or st.session_state.get('query_table_changed'):
                st.session_state.text2sql_pipeline = EnhancedText2SQLPipeline(
                    st.session_state.db_manager, 
                    query_table if query_table else None
                )
                st.session_state.query_table_changed = False
            
            # Check if table selection changed
            if st.session_state.get('last_query_table') != query_table:
                st.session_state.last_query_table = query_table
                st.session_state.query_table_changed = True
                st.rerun()
            
            # Query interface
            question = st.text_area("Enter your question in natural language:", 
                                   placeholder="例: KICで発生したトラブルの件数を教えてください。")
            
            if st.button("🔍 Query"):
                if question:
                    with st.spinner("Generating SQL and executing query..."):
                        try:
                            result = st.session_state.text2sql_pipeline.query(question)
                            
                            # Display results
                            st.subheader("🤔 Your Question")
                            st.write(result["question"])
                            
                            st.subheader("🔍 Generated SQL")
                            st.code(result["sql_query"], language="sql")
                            
                            st.subheader("📊 Query Results")
                            if not result["sql_results"].empty:
                                st.dataframe(result["sql_results"], hide_index=True)
                            else:
                                st.info("No results found.")
                            
                            st.subheader("🧠 AI Response")
                            st.write(result["final_answer"])
                            
                        except Exception as e:
                            st.error(f"Error processing query: {str(e)}")
                else:
                    st.warning("Please enter a question.")
    
    else:
        st.info("👈 Please connect to the database using the sidebar.")

if __name__ == "__main__":
    main() 