"""
Separate Text-to-SQL Pipeline Approach
"""

from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.core import Settings
from sqlalchemy import create_engine, text
import pandas as pd
import streamlit as st

class Text2SQLPipeline:
    """Separate pipeline for text-to-SQL conversion and response generation."""
    
    def __init__(self):
        # Configuration - read from secrets if available
        try:
            api_key = st.secrets["openai"]["api_key"]
            azure_endpoint = st.secrets["openai"]["azure_endpoint"]
            api_version = st.secrets["openai"]["api_version"]
        except (KeyError, AttributeError):
            raise ValueError("OpenAI API key not found in secrets")

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

        # SQLAlchemy engine
        self.engine = create_engine("postgresql://testuser:testpass@localhost:5432/testdb")
        
        # Get table schema for context
        self.schema_info = self._get_schema_info()
    
    def _get_schema_info(self) -> str:
        """Get table schema information for SQL generation."""
        schema_query = """
        SELECT 
            column_name, 
            data_type, 
            is_nullable,
            column_default
        FROM information_schema.columns 
        WHERE table_name = 'trouble_reports'
        ORDER BY ordinal_position;
        """
        
        with self.engine.connect() as conn:
            result = conn.execute(text(schema_query))
            columns = result.fetchall()
        
        schema_text = "Table: trouble_reports\nColumns:\n"
        for col in columns:
            schema_text += f"- {col[0]} ({col[1]})\n"
        
        return schema_text
    
    def natural_language_to_sql(self, question: str) -> str:
        """Convert natural language question to SQL query."""
        
        prompt = f"""
        You are an expert SQL query generator. Given a natural language question, generate a PostgreSQL query.

        Database Schema:
        {self.schema_info}

        Question: {question}

        Instructions:
        - Generate only the SQL query, no explanation
        - Use proper PostgreSQL syntax
        - Return only the SELECT statement
        - For Japanese text, use appropriate filtering

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
    
    def execute_sql(self, sql_query: str) -> pd.DataFrame:
        """Execute SQL query and return results as DataFrame."""
        try:
            with self.engine.connect() as conn:
                result = pd.read_sql_query(sql_query, conn)
            return result
        except Exception as e:
            print(f"Error executing SQL: {e}")
            return pd.DataFrame()
    
    def generate_response(self, question: str, sql_query: str, sql_results: pd.DataFrame) -> str:
        """Generate natural language response from SQL results."""
        
        # Convert DataFrame to text for the prompt
        if sql_results.empty:
            results_text = "No results found."
        else:
            results_text = sql_results.to_string(index=False)
        
        prompt = f"""
        You are an assistant that explains database query results in natural language.

        Original Question: {question}
        SQL Query Used: {sql_query}
        Query Results:
        {results_text}

        Instructions:
        - Provide a clear, comprehensive answer in Japanese
        - Summarize the key findings from the data
        - If there are multiple records, organize them clearly
        - Include relevant details from the results

        Response:
        """
        
        response = self.llm.complete(prompt)
        return response.text.strip()
    
    def query(self, question: str) -> dict:
        """Full pipeline: question -> SQL -> results -> response."""
        
        print(f"🤔 Question: {question}")
        
        # Step 1: Convert to SQL
        sql_query = self.natural_language_to_sql(question)
        print(f"🔍 Generated SQL: {sql_query}")
        
        # Step 2: Execute SQL
        sql_results = self.execute_sql(sql_query)
        print(f"📊 Query returned {len(sql_results)} rows")
        
        # Step 3: Generate response
        final_response = self.generate_response(question, sql_query, sql_results)
        print(f"🧠 Final Answer:\n{final_response}")
        
        return {
            "question": question,
            "sql_query": sql_query,
            "sql_results": sql_results,
            "final_answer": final_response
        }

def main():
    """Test the Text-to-SQL pipeline."""
    pipeline = Text2SQLPipeline()
    
    # Test questions
    questions = [
        "KICで発生したトラブルの件数とその内容を教えてください。",
        "2021年に発生したトラブルを教えてください。",
        "最新のトラブル報告は何ですか？"
    ]
    
    for question in questions:
        print("=" * 60)
        result = pipeline.query(question)
        print("\n")

if __name__ == "__main__":
    main() 