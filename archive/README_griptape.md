# Database Management & Griptape SQL Query Engine

This is an enhanced version of the database application that uses **Griptape AI** for natural language to SQL conversion, with LlamaIndex as a fallback option.

## Key Features

### 🤖 Griptape AI Integration
- **Natural Language to SQL**: Uses Griptape's Pipeline and PromptTask for intelligent SQL generation
- **Context-Aware**: Loads table schemas and sample data for better query generation
- **PostgreSQL Optimized**: Generates PostgreSQL-specific syntax
- **Japanese Text Support**: Handles Japanese text in queries and responses
- **Intelligent Formatting**: AI-powered result interpretation and formatting

### 🔄 Dual Engine Support
- **Primary**: Griptape AI (recommended)
- **Fallback**: LlamaIndex NLSQLTableQueryEngine (if available)
- **Automatic Detection**: Gracefully handles missing dependencies

### 📊 Enhanced Database Operations
- **Griptape SQL Driver**: Uses `SqlDriver` for database operations
- **Schema Intelligence**: Automatic table schema loading with sample data
- **Fallback Execution**: Falls back to SQLAlchemy if Griptape execution fails

## Installation

Install the required dependencies:

```bash
pip install -r requirements_griptape.txt
```

## Key Differences from Original

| Feature | Original (LlamaIndex) | New (Griptape) |
|---------|----------------------|----------------|
| SQL Generation | NLSQLTableQueryEngine | Griptape Pipeline + PromptTask |
| Database Driver | SQLAlchemy only | Griptape SqlDriver + SQLAlchemy fallback |
| AI Framework | LlamaIndex | Griptape (with LlamaIndex fallback) |
| Japanese Support | Basic | Enhanced with AI formatting |
| Result Formatting | Built-in | AI-powered interpretation |
| Engine Selection | Single | Dual with user choice |

## How It Works

### 1. **Schema Loading**
```python
def _load_table_schemas(self):
    # Loads table structures and sample data for context
    for table in tables_to_load:
        schema_text = f"Table: {table}\nColumns:\n..."
        sample_data = self.db_manager.get_table_data(table, 3)
        self.table_schemas[table] = schema_text
```

### 2. **SQL Generation with Griptape**
```python
def _generate_sql_with_ai(self, question: str) -> str:
    pipeline = Pipeline()
    pipeline.add_tasks(
        PromptTask(
            rules=[
                Rule("Generate ONLY valid PostgreSQL SQL queries"),
                Rule("Use proper PostgreSQL syntax and functions"),
                # ... more rules
            ],
        ),
    )
    # Includes full schema context in prompt
    return pipeline.run(prompt)
```

### 3. **Execution with Fallback**
```python
def _execute_sql_with_griptape(self, sql_query: str):
    try:
        # Try Griptape driver first
        return self.db_manager.griptape_driver.execute_query(sql_query)
    except Exception:
        # Fallback to SQLAlchemy
        return self.db_manager.execute_query(sql_query)
```

### 4. **AI-Powered Result Formatting**
```python
def _format_result_with_ai(self, sql_query: str, raw_result, question: str) -> str:
    # Uses Griptape AI to interpret and format results
    # Provides conversational, context-aware answers
```

## Usage

1. **Run the Application**:
   ```bash
   streamlit run main_griptape.py
   ```

2. **Select Query Engine**:
   - Choose "Griptape AI" for the enhanced experience
   - Choose "LlamaIndex (fallback)" if needed

3. **Select Tables**: Choose specific tables or use all tables

4. **Ask Questions**: Natural language queries in English or Japanese

## Example Queries

- "How many records are in each table?"
- "Show me the top 10 entries by date"
- "KICで発生したトラブルの件数を教えてください" (Japanese)
- "What's the average value in the price column?"

## Benefits of Griptape Integration

1. **More Control**: Explicit rules and pipeline structure
2. **Better Context**: Rich schema information with sample data
3. **Enhanced Flexibility**: Easier to customize AI behavior
4. **Robust Fallbacks**: Multiple levels of fallback (engine, execution, formatting)
5. **Japanese Support**: Better handling of multilingual content
6. **Debugging**: Clear separation of SQL generation, execution, and formatting

## Configuration

Ensure your Streamlit secrets include:

```toml
[openai]
api_key = "your-openai-api-key"
azure_endpoint = "your-azure-endpoint"  # if using Azure
api_version = "your-api-version"  # if using Azure

[database]
password = "your-database-password"
```

## Dependencies

- **Griptape**: AI framework and SQL drivers
- **Streamlit**: Web interface
- **SQLAlchemy**: Database operations
- **Pandas**: Data manipulation
- **LlamaIndex**: Fallback query engine (optional)

## Troubleshooting

1. **Griptape Import Errors**: Install with `pip install griptape[drivers-sql]`
2. **Database Connection**: Check PostgreSQL driver installation
3. **AI Generation Issues**: Check OpenAI API configuration
4. **Japanese Text**: Ensure PostgreSQL supports Unicode 