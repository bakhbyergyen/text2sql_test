"""
Database Application with Table Management and NLSQLTableQueryEngine Integration
"""

import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text, inspect
import os
os.environ["OPENAI_API_KEY"] = st.secrets["openai"]["api_key"]
from urllib.parse import quote_plus
import re
import json
from griptape.rules import Rule
from griptape.structures import Pipeline
from griptape.tasks import PromptTask

# LlamaIndex imports for SQL query engine
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.core import SQLDatabase, Settings
from llama_index.core.query_engine import NLSQLTableQueryEngine

class DatabaseManager:
    """Manage database connections and operations."""
    
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.engine = None
        self.connect()
    
    def translate_column_names(self, japanese_columns: list) -> dict:
        """Translate Japanese column names to English using Griptape AI."""
        try:
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

class SQLQueryEngine:
    """Enhanced SQL Query Engine using NLSQLTableQueryEngine."""
    
    def __init__(self, db_manager: DatabaseManager, selected_tables: list = None):
        self.db_manager = db_manager
        self.selected_tables = selected_tables
        self.llm = None
        self.sql_engine = None
        self._setup_llm()
        self._setup_sql_engine()
    
    def _setup_llm(self):
        """Setup the LLM from secrets."""
        try:
            api_key = st.secrets["openai"]["api_key"]
            azure_endpoint = st.secrets["openai"]["azure_endpoint"]
            api_version = st.secrets["openai"]["api_version"]
        except KeyError as e:
            st.error(f"❌ Missing OpenAI secret key: {e}")
            raise Exception(f"Missing OpenAI configuration in secrets: {e}")

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
    
    def _setup_sql_engine(self):
        """Setup the SQL database and query engine."""
        try:
            # Determine which tables to include
            if self.selected_tables:
                include_tables = self.selected_tables
            else:
                # Include all tables if none specified
                include_tables = self.db_manager.get_tables()
            
            # Create SQLDatabase
            sql_db = SQLDatabase(self.db_manager.engine, include_tables=include_tables)
            
            # Create SQL query engine with enhanced prompting
            self.sql_engine = NLSQLTableQueryEngine(
                sql_database=sql_db,
                verbose=True,
            )
            
            st.success(f"✅ SQL Query Engine initialized with tables: {', '.join(include_tables)}")
            
        except Exception as e:
            st.error(f"❌ Error setting up SQL engine: {str(e)}")
            raise
    
    def query(self, question: str):
        """Query using the NLSQLTableQueryEngine."""
        if not self.sql_engine:
            raise Exception("SQL engine not initialized")
        
        try:
            st.info(f"🤔 Processing question: {question}")
            
            # Execute the query
            response = self.sql_engine.query(question)
            
            # Extract information from response
            result = {
                "question": question,
                "sql_query": response.metadata.get('sql_query', 'N/A'),
                "final_answer": str(response),
                "response_object": response
            }
            
            return result
            
        except Exception as e:
            st.error(f"❌ Error processing query: {str(e)}")
            raise

def main():
    """Main Streamlit application."""
    
    st.set_page_config(
        page_title="Database Management & SQL Query Engine",
        page_icon="🗄️",
        layout="wide"
    )
    
    st.title("🗄️ Database Management & SQL Query Engine")
    st.markdown("Connect to your database, manage tables, upload Excel files, and query with natural language using LlamaIndex!")
    
    # Initialize session state
    if 'db_manager' not in st.session_state:
        st.session_state.db_manager = None
    if 'selected_tables' not in st.session_state:
        st.session_state.selected_tables = []
    
    # Auto-connect to database on startup
    if st.session_state.db_manager is None:
        try:
            with st.spinner("🔌 Connecting to database..."):
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
            st.stop()
        except Exception as e:
            st.error(f"❌ Connection error: {e}")
            st.stop()
    
    # Main content
    if st.session_state.db_manager:
        
        # Create tabs
        tab1, tab2, tab3 = st.tabs(["📊 Table Management", "📤 Upload Excel", "🤖 SQL Query Engine"])
        
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
            st.header("🤖 SQL Query Engine")
            st.markdown("Powered by LlamaIndex NLSQLTableQueryEngine")
            
            # Table selection for queries
            tables = st.session_state.db_manager.get_tables()
            
            if tables:
                # Simple table selection
                st.subheader("📋 Table Selection")
                col1, col2 = st.columns([2, 1])
                with col1:
                    selected_tables = st.multiselect(
                        "Select tables for querying:", 
                        tables,
                        default=[],
                        key="query_tables",
                        help="Select specific tables or leave empty to use all tables"
                    )
                
                with col2:
                    use_all_tables = st.button("🔄 Use All Tables")
                    if use_all_tables:
                        st.session_state.query_tables = []
                        st.session_state.use_all_tables_clicked = True
                        st.rerun()
                
                # Check if table selection changed
                if st.session_state.get('last_selected_tables') != selected_tables:
                    st.session_state.last_selected_tables = selected_tables
                    # Clear the existing engine when tables change
                    if 'sql_query_engine' in st.session_state:
                        del st.session_state.sql_query_engine
                
                # Determine if user has made a choice
                user_made_choice = (
                    selected_tables or 
                    st.session_state.get('use_all_tables_clicked', False)
                )
                
                if user_made_choice:
                    # Show current selection and initialize engine
                    if selected_tables:
                        st.info(f"🎯 **Selected tables**: {', '.join(selected_tables)}")
                        tables_to_use = selected_tables
                    else:
                        st.info(f"🌐 **Using all tables**: {', '.join(tables)}")
                        tables_to_use = None
                    
                    # Initialize engine if not exists
                    if 'sql_query_engine' not in st.session_state:
                        try:
                            with st.spinner("🔧 Initializing SQL Query Engine..."):
                                st.session_state.sql_query_engine = SQLQueryEngine(
                                    st.session_state.db_manager, 
                                    tables_to_use
                                )
                            st.success("✅ Query Engine ready!")
                        except Exception as e:
                            st.error(f"❌ Error initializing SQL Query Engine: {str(e)}")
                            st.error("Please check your OpenAI configuration in secrets.")
                            st.stop()
                    
                    # Query interface
                    st.divider()
                    st.subheader("💬 Ask Your Question")
                    
                    question = st.text_area(
                        "Enter your question in natural language:", 
                        placeholder="例: KICで発生したトラブルの件数を教えてください。\nExample: How many trouble reports are there from KIC?",
                        height=100,
                        key="question_input"
                    )
                    
                    # Use form to prevent premature rerendering
                    with st.form("query_form"):
                        submit_button = st.form_submit_button("🔍 Execute Query", type="primary")
                    
                    if submit_button and question.strip():
                        with st.spinner("🤔 Processing your question..."):
                            try:
                                result = st.session_state.sql_query_engine.query(question)
                                
                                # Display results
                                st.success("✅ Query executed successfully!")
                                
                                # Create tabs for results
                                result_tab1, result_tab2 = st.tabs(["📊 Answer & SQL", "🔍 Details"])
                                
                                with result_tab1:
                                    col1, col2 = st.columns([1, 1])
                                    
                                    with col1:
                                        st.subheader("🧠 AI Answer")
                                        st.write(result["final_answer"])
                                    
                                    with col2:
                                        st.subheader("🔍 Generated SQL")
                                        st.code(result["sql_query"], language="sql")
                                
                                with result_tab2:
                                    st.subheader("❓ Your Question")
                                    st.write(result["question"])
                                    
                                    # Additional metadata if available
                                    if hasattr(result["response_object"], 'source_nodes'):
                                        st.subheader("📚 Source Information")
                                        for i, node in enumerate(result["response_object"].source_nodes):
                                            with st.expander(f"Source {i+1}"):
                                                st.write(node.text)
                            
                            except Exception as e:
                                st.error(f"❌ Error processing query: {str(e)}")
                                st.error("Please check your question and try again.")
                    
                    # Show help
                    with st.expander("💡 Query Tips"):
                        st.markdown("""
                        **Example questions you can ask:**
                        - How many records are in the table?
                        - What are the unique values in column X?
                        - Show me the top 10 entries by date
                        - Count records where condition Y is true
                        - What's the average/sum/max of column Z?
                        """)
                else:
                    st.info("📝 **Please select tables first to enable the query interface.**")
                    st.markdown("Choose specific tables from the dropdown above, or click 'Use All Tables' to proceed.")
                
            else:
                st.info("📋 No tables found in the database. Please upload some data first.")
    
    else:
        st.error("❌ Failed to connect to database. Please check your configuration.")
        st.stop()

if __name__ == "__main__":
    main() 