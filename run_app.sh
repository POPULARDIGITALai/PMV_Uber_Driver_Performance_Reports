#!/bin/bash

# Driver Performance Dashboard Launcher
echo "Starting Driver Performance Dashboard..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Run the Streamlit app
echo "Launching Dashboard..."
streamlit run app.py --server.port 8501 --server.address 0.0.0.0

echo "Dashboard is running at http://localhost:8501"