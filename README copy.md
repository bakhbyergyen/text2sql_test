# Database Management & Text2SQL Application

A comprehensive database application that connects to Supabase PostgreSQL databases, manages tables, uploads Excel files, and provides natural language querying capabilities using Azure OpenAI.

## Features

### 🔌 Database Connection
- Connect to Supabase PostgreSQL databases
- Secure password input with connection status display
- Automatic connection testing

### 📊 Table Management
- List all tables in the public schema
- View table schema information (columns, types, nullable)
- Preview table data with sample rows
- Interactive table selection

### 📤 Excel File Upload
- Upload Excel files (.xlsx, .xls) to database tables
- **Japanese Column Support**: Automatic detection and handling of Japanese column names
- **AI Translation**: Optional translation of Japanese column names to English using Griptape AI
- **Smart Column Cleaning**: Preserves Japanese characters while ensuring database compatibility
- Flexible upload options (replace, append, fail if exists)
- Preview Excel data before upload
- Duplicate column name detection and resolution

### 🤖 Text2SQL Query
- Natural language to SQL conversion using Azure OpenAI
- Support for Japanese and English queries
- Table-specific or database-wide querying
- Complete query pipeline with results and AI-generated responses

## Setup Instructions

1. **Navigate to the application directory**
   ```bash
   cd sql_query/database_app
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Secrets**
   ```bash
   # Copy the example secrets file
   cp .streamlit/secrets.toml.example .streamlit/secrets.toml
   
   # Edit the secrets file with your credentials
   # - Database password
   # - Azure OpenAI API key and endpoint
   # - OpenAI API key for Griptape translation
   ```

4. **Run the Application**
   ```bash
   streamlit run main.py
   ```
   Or use the convenient script:
   ```bash
   ./run_app.sh
   ```

## Usage Guide

### Step 1: Connect to Database
1. Ensure your credentials are configured in `.streamlit/secrets.toml`
2. Click "Connect to Database" in the sidebar
3. Verify the connection status (🟢 Connected)

### Step 2: Manage Tables
1. Go to the "Table Management" tab
2. Select a table from the dropdown
3. View table schema and sample data
4. Use this information to understand your data structure

### Step 3: Upload Excel Files
1. Go to the "Upload Excel" tab
2. Choose an Excel file using the file uploader
3. Preview the data to ensure it's correct
4. Enter a table name for the upload
5. Choose upload behavior (replace/append/fail)
6. **For Japanese Excel files**:
   - The app automatically detects Japanese column names
   - Choose whether to translate to English or keep Japanese names
   - Translation uses Griptape AI for accurate results
7. Click "Upload to Database"

### Step 4: Query with Natural Language
1. Go to the "Text2SQL Query" tab
2. Optionally select a specific table to focus queries
3. Enter your question in natural language (supports Japanese)
4. Click "Query" to see:
   - Generated SQL query
   - Query results
   - AI-generated natural language response

## Example Queries

- "Show me all data from the users table"
- "How many records are in the sales table?"
- "Find the latest 10 entries"
- "KICで発生したトラブルの件数を教えてください。" (Japanese)

## Japanese Excel File Support

### Column Name Examples
**Original Japanese → Cleaned/Translated:**
- `名前` → Keeps as `名前` (Japanese) or translates to `name` (English)
- `年齢` → Keeps as `年齢` (Japanese) or translates to `age` (English)
- `電話番号` → Keeps as `電話番号` (Japanese) or translates to `phone_number` (English)
- `（備考）` → Cleaned to `_備考_` (Japanese) or translates to `notes` (English)

### Features
- **Unicode Support**: PostgreSQL fully supports Japanese characters in column names
- **AI Translation**: Griptape AI provides contextually accurate translations
- **Flexible Options**: Choose between Japanese preservation or English translation
- **Error Handling**: Automatic fallback if translation fails

## Technical Details

### Architecture
- **Frontend**: Streamlit web application
- **Database**: PostgreSQL (Supabase) with Unicode/Japanese support
- **AI**: 
  - Azure OpenAI GPT-4o for text-to-SQL conversion
  - Griptape AI for Japanese column name translation
- **Data Processing**: Pandas for Excel file handling with Japanese character support

### Key Components
- `DatabaseManager`: Handles all database operations
- `EnhancedText2SQLPipeline`: Extends the original pipeline with dynamic table selection
- Streamlit UI: Provides interactive web interface

### Security
- **Streamlit Secrets**: All credentials stored in secure secrets file
- **No Hardcoded Credentials**: No sensitive data in source code
- **Git Ignored**: secrets.toml automatically excluded from version control
- **SQL Injection Protection**: SQLAlchemy provides secure database connections
- **Environment Isolation**: Credentials loaded at runtime from secrets

## Troubleshooting

### Common Issues
1. **Connection Failed**: 
   - Check `.streamlit/secrets.toml` exists and has correct database password
   - Verify network connection to Supabase
2. **Missing Secrets**: 
   - Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml`
   - Fill in your actual credentials
3. **Import Errors**: Ensure all dependencies are installed
4. **Excel Upload Failed**: Check file format and column names
5. **Translation Errors**: Verify Griptape OpenAI API key in secrets
6. **Query Errors**: Verify Azure OpenAI configuration in secrets

### Dependencies
Make sure you have all required packages installed from `requirements.txt`

## File Structure
```
sql_query/database_app/
├── main.py                        # Main application file
├── requirements.txt               # Python dependencies
├── run_app.sh                    # Convenient startup script
├── README.md                     # This documentation
├── .gitignore                    # Git ignore file (excludes secrets)
└── .streamlit/
    ├── secrets.toml              # Your actual credentials (not in git)
    └── secrets.toml.example      # Example secrets configuration
```

## Notes
- The application uses the existing Text2SQL pipeline from the parent directory
- Excel column names are automatically cleaned for database compatibility
- The application focuses on the public schema in PostgreSQL
- Session state maintains connection and pipeline instances across interactions 