#!/bin/bash

# This script starts both the API and the frontend application with fixed absolute paths

# Kill any running uvicorn or streamlit processes
echo "Killing any running uvicorn or streamlit processes..."
pkill -f "uvicorn|streamlit" || true

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

# Copy the model to ensure it's in both places
echo "Ensuring model file is accessible..."
mkdir -p api/models
cp models/fraud_detection_model.pkl api/models/

# Start the API server in the background from project root to maintain correct relative paths
echo "Starting the FastAPI server on port $API_PORT..."
cd "$(dirname "$0")"
python -m uvicorn api.main:app --host 0.0.0.0 --port $API_PORT &
API_PID=$!

# Give API time to start
sleep 3

# Start the Streamlit app
echo "Starting the Streamlit dashboard on port $STREAMLIT_PORT..."
cd frontend
streamlit run app.py --server.port=$STREAMLIT_PORT &
STREAMLIT_PID=$!

echo ""
echo "====================================================="
echo "API running at: http://localhost:$API_PORT"
echo "Dashboard running at: http://localhost:$STREAMLIT_PORT"
echo "====================================================="
echo "Press Ctrl+C to stop both services"
echo ""

# Wait for user interrupt
trap "kill $API_PID $STREAMLIT_PID 2>/dev/null" EXIT
wait
