import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime

# Load the model and scalers
try:
    # Try different possible paths for the model file
    model_paths = [
        'models/fraud_detection_model.pkl',  # Relative from API directory
        '../models/fraud_detection_model.pkl',  # One directory up
        '/Users/chethan/Desktop/credit-card-fraud/models/fraud_detection_model.pkl'  # Absolute path
    ]
    
    model_loaded = False
    for path in model_paths:
        try:
            model_info = joblib.load(path)
            model = model_info['model']
            feature_scaler = model_info['feature_scaler']
            time_amount_scaler = model_info['time_amount_scaler']
            feature_names = model_info['feature_names']
            model_loaded = True
            print(f"Model loaded successfully from: {path}")
            break
        except Exception as e:
            print(f"Failed to load model from {path}: {e}")
    
    if not model_loaded:
        raise Exception("Could not load model from any known paths")
            
except Exception as e:
    print(f"Error loading model: {e}")
    # We'll use None as placeholder since the model will be trained and saved through the notebooks
    model, feature_scaler, time_amount_scaler, feature_names = None, None, None, None

# Define the API app
app = FastAPI(title="Credit Card Fraud Detection API",
              description="API for detecting fraudulent credit card transactions",
              version="1.0.0")

# Define the input data model
class Transaction(BaseModel):
    time: float = None  # Time since first transaction in seconds
    amount: float  # Transaction amount
    v1: float = 0.0
    v2: float = 0.0
    v3: float = 0.0
    v4: float = 0.0
    v5: float = 0.0
    v6: float = 0.0
    v7: float = 0.0
    v8: float = 0.0
    v9: float = 0.0
    v10: float = 0.0
    v11: float = 0.0
    v12: float = 0.0
    v13: float = 0.0
    v14: float = 0.0
    v15: float = 0.0
    v16: float = 0.0
    v17: float = 0.0
    v18: float = 0.0
    v19: float = 0.0
    v20: float = 0.0
    v21: float = 0.0
    v22: float = 0.0
    v23: float = 0.0
    v24: float = 0.0
    v25: float = 0.0
    v26: float = 0.0
    v27: float = 0.0
    v28: float = 0.0

class PredictionResponse(BaseModel):
    transaction_id: str
    is_fraud: bool
    fraud_probability: float
    timestamp: str
    risk_level: str
    suggested_action: str

# Reference time for calculating the 'time' feature if not provided
REFERENCE_TIME = datetime.now().timestamp()

@app.get("/")
async def root():
    return {"message": "Credit Card Fraud Detection API"}

@app.post("/predict", response_model=PredictionResponse)
async def predict_fraud(transaction: Transaction):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please train the model first.")
    
    # Generate transaction ID
    transaction_id = f"tx_{int(datetime.now().timestamp() * 1000)}"
    
    # Calculate time if not provided
    if transaction.time is None:
        transaction.time = datetime.now().timestamp() - REFERENCE_TIME
    
    # Create a DataFrame with the transaction features
    features = {}
    
    # Handle time and amount separately
    time_amount = np.array([[transaction.time, transaction.amount]])
    scaled_time_amount = time_amount_scaler.transform(time_amount)
    
    # Extract V features
    v_features = []
    for i in range(1, 29):
        v_features.append(getattr(transaction, f'v{i}'))
    v_features_array = np.array([v_features])
    scaled_v_features = feature_scaler.transform(v_features_array)
    
    # Combine all features
    combined_features = np.hstack((scaled_time_amount, scaled_v_features))
    
    # Make prediction
    fraud_probability = model.predict_proba(combined_features)[0, 1]
    is_fraud = model.predict(combined_features)[0] == 1
    
    # Determine risk level and suggested action
    if fraud_probability < 0.3:
        risk_level = "Low"
        suggested_action = "Approve"
    elif fraud_probability < 0.7:
        risk_level = "Medium"
        suggested_action = "Review"
    else:
        risk_level = "High"
        suggested_action = "Block and contact customer"
    
    return PredictionResponse(
        transaction_id=transaction_id,
        is_fraud=is_fraud,
        fraud_probability=float(fraud_probability),
        timestamp=datetime.now().isoformat(),
        risk_level=risk_level,
        suggested_action=suggested_action
    )

@app.get("/health")
async def health_check():
    return {
        "status": "healthy" if model is not None else "model not loaded",
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
