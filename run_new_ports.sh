#!/bin/bash

# This script starts both the API and the frontend application with alternative ports

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
API_PORT=8005
STREAMLIT_PORT=8506

if port_in_use $API_PORT; then
  echo "Port $API_PORT is already in use. Please free up this port for the API."
  exit 1
fi

if port_in_use $STREAMLIT_PORT; then
  echo "Port $STREAMLIT_PORT is already in use. Please free up this port for the Streamlit app."
  exit 1
fi

# Start the API server in the background
echo "Starting the FastAPI server on port $API_PORT..."
cd api
uvicorn main:app --host 0.0.0.0 --port $API_PORT &
API_PID=$!
cd ..

# Give API time to start
sleep 2

# Start the Streamlit app
echo "Starting the Streamlit dashboard on port $STREAMLIT_PORT..."
cd frontend
streamlit run app.py --server.port=$STREAMLIT_PORT

# When the Streamlit app is closed, also close the API server
kill $API_PID
