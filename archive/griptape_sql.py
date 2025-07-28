from griptape.drivers.sql.sql_driver import SqlDriver
from griptape.structures import Pipeline
from griptape.tasks import PromptTask
from griptape.rules import Rule
import pandas as pd

class LlamaIndexStyleGriptapeEngine:
    """Uses LlamaIndex logic with Griptape SqlDriver"""
    
    def __init__(self, connection_string: str, schema_name: str = "public"):
        self.driver = SqlDriver(engine_url=connection_string)
        self.schema_name = schema_name
        
        # LlamaIndex prompts
        self.text_to_sql_prompt = """Given an input question, first create a syntactically correct postgresql query to run, then look at the results of the query and return the answer. You can order the results by a relevant column to return the most interesting examples in the database.

        Never query for all the columns from a specific table, only ask for a few relevant columns given the question.

        Pay attention to use only the column names that you can see in the schema description. Be careful to not query for columns that do not exist. Pay attention to which column is in which table. Also, qualify column names with the table name when needed. You are required to use the following format, each taking one line:

        Question: Question here
        SQLQuery: SQL Query to run
        SQLResult: Result of the SQLQuery
        Answer: Final answer here

        Only use tables listed below.
        {schema}

        Question: {query_str}
        SQLQuery: """

        self.response_synthesis_prompt = """Given an input question, synthesize a response from the query results.
        Query: {query_str}
        SQL: {sql_query}
        SQL Response: {context_str}
        Response: """

    def get_table_schema(self, table_name: str) -> str:
        """Get schema using Griptape SqlDriver"""
        return self.driver.get_table_schema(table_name=table_name)
    
    def generate_sql(self, question: str, table_schemas: str) -> str:
        """Step 1: Generate SQL using LlamaIndex prompt logic"""
        pipeline = Pipeline()
        pipeline.add_tasks(
            PromptTask(
                rules=[
                    Rule("You are an expert SQL developer"),
                    Rule("Follow the exact format: Question/SQLQuery/SQLResult/Answer"),
                    Rule("Return ONLY the SQL query part after 'SQLQuery:'"),
                    Rule("Use proper PostgreSQL syntax"),
                    Rule("Be careful with column names and table names"),
                ]
            )
        )
        
        prompt = self.text_to_sql_prompt.format(
            schema=table_schemas,
            query_str=question
        )
        
        result = pipeline.run(prompt)
        sql_query = self.parse_sql_from_response(str(result))
        return sql_query
    
    def parse_sql_from_response(self, response: str) -> str:
        """Parse SQL using LlamaIndex's exact logic"""
        # Find and remove SQLResult part
        sql_result_start = response.find("SQLResult:")
        if sql_result_start != -1:
            response = response[:sql_result_start]
        
        # Extract SQL query from various formats
        sql_query_start = response.find("SQLQuery:")
        if sql_query_start != -1:
            response = response[sql_query_start:]
            response = response.replace("SQLQuery:", "")
        
        # Handle markdown code blocks
        sql_markdown_start = response.find("```sql")
        if sql_markdown_start != -1:
            response = response.replace("```sql", "")
        response = response.replace("```", "")
        
        # Handle semicolon termination
        semi_colon = response.find(";")
        if semi_colon != -1:
            response = response[: semi_colon + 1]
        
        # Replace escaped single quotes
        response = response.replace("\\'", "''")
        return response.strip()
    
    def execute_sql(self, sql_query: str) -> pd.DataFrame:
        """Execute SQL using Griptape SqlDriver"""
        try:
            result = self.driver.execute_query(sql_query)
            # Convert to DataFrame
            if hasattr(result, 'fetchall'):
                data = result.fetchall()
                columns = result.keys()
                return pd.DataFrame(data, columns=columns)
            return pd.DataFrame(result)
        except Exception as e:
            raise Exception(f"SQL execution error: {str(e)}")
    
    def synthesize_answer(self, question: str, sql_query: str, sql_results: pd.DataFrame) -> str:
        """Step 2: Generate answer using LlamaIndex synthesis logic"""
        if sql_results.empty:
            results_text = "No results found."
        else:
            results_text = sql_results.to_string(index=False)
        
        pipeline = Pipeline()
        pipeline.add_tasks(
            PromptTask(
                rules=[
                    Rule("You are an assistant that explains database results"),
                    Rule("Provide clear, comprehensive answers"),
                    Rule("Summarize key findings from the data"),
                    Rule("Use the exact format from the synthesis prompt")
                ]
            )
        )
        
        prompt = self.response_synthesis_prompt.format(
            query_str=question,
            sql_query=sql_query,
            context_str=results_text
        )
        
        result = pipeline.run(prompt)
        return str(result).strip()
    
    def query(self, question: str, table_names: list) -> dict:
        """Complete pipeline: Question → SQL → Execute → Answer"""
        try:
            # Get schemas for all tables
            table_schemas = []
            for table_name in table_names:
                schema = self.get_table_schema(table_name)
                table_schemas.append(f"Table: {table_name}\n{schema}")
            
            combined_schema = "\n\n".join(table_schemas)
            
            # Step 1: Generate SQL (LlamaIndex logic)
            sql_query = self.generate_sql(question, combined_schema)
            
            # Step 2: Execute SQL (Griptape SqlDriver)
            sql_results = self.execute_sql(sql_query)
            
            # Step 3: Generate Answer (LlamaIndex logic)
            final_answer = self.synthesize_answer(question, sql_query, sql_results)
            
            return {
                "question": question,
                "sql_query": sql_query,
                "sql_results": sql_results,
                "final_answer": final_answer,
                "success": True
            }
            
        except Exception as e:
            return {
                "question": question,
                "error": str(e),
                "success": False
            }

# Usage
engine = LlamaIndexStyleGriptapeEngine(
    connection_string="postgresql://user:pass@host:port/db"
)

result = engine.query(
    question="KICで発生したトラブルの件数を教えてください",
    table_names=["trouble_reports"]
)

print(f"Question: {result['question']}")
print(f"SQL: {result['sql_query']}")
print(f"Answer: {result['final_answer']}")