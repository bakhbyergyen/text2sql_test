#!/bin/bash

# Database Management & Text2SQL Application Launcher

echo "🚀 Starting Database Management & Text2SQL Application..."
echo "📋 Make sure you have installed the dependencies: pip install -r requirements.txt"
echo "🔗 The application will open in your browser at http://localhost:8501"
echo ""

# Change to the directory where the script is located
cd "$(dirname "$0")"

# Run the Streamlit application
streamlit run main.py --server.port 8501 --server.address localhost 