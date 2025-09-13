#!/bin/bash

# This script starts both the API and the frontend application

# Check if Python virtual environment exists
if [ ! -d ".venv" ]; then
  echo "Virtual environment not found. Creating one..."
  python -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
else
  source .venv/bin/activate
fi

# Function to check if a port is in use
port_in_use() {
  lsof -i:$1 >/dev/null 2>&1
  return $?
}

# Check if ports are available
if port_in_use 8000; then
  echo "Port 8000 is already in use. Please free up this port for the API."
  exit 1
fi

if port_in_use 8501; then
  echo "Port 8501 is already in use. Please free up this port for the Streamlit app."
  exit 1
fi

# Start the API server in the background
echo "Starting the FastAPI server..."
cd api
uvicorn main:app --host 0.0.0.0 --port 8000 &
API_PID=$!
cd ..

# Give API time to start
sleep 2

# Start the Streamlit app
echo "Starting the Streamlit dashboard..."
cd frontend
streamlit run app.py

# When the Streamlit app is closed, also close the API server
kill $API_PID
