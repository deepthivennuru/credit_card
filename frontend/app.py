import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import json
from datetime import datetime
import time
import random
from io import StringIO

# Set page configuration
st.set_page_config(
    page_title="Credit Card Fraud Detection Dashboard",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
API_URL = "http://localhost:8005"  # Updated to match the port in run_new_ports.sh

# Custom CSS
st.markdown("""
<style>
    .fraud {
        color: #FF4B4B;
        font-weight: bold;
    }
    .normal {
        color: #00CC96;
        font-weight: bold;
    }
    .dashboard-title {
        font-size: 42px;
        font-weight: bold;
        color: #1E88E5;
        margin-bottom: 20px;
    }
    .section-title {
        font-size: 24px;
        font-weight: bold;
        color: #333333;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    .card {
        padding: 20px;
        border-radius: 5px;
        margin-bottom: 20px;
        background-color: #FFFFFF;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.image("https://img.icons8.com/color/96/000000/bank-cards.png", width=100)
st.sidebar.title("Fraud Detection")
page = st.sidebar.selectbox("Choose a page", ["Dashboard", "Simulate Transactions", "About"])

# Load sample data
@st.cache_data
def load_sample_data():
    # This is a placeholder. In a real app, you'd connect to a database
    # or load from a file with actual transaction data
    try:
        df = pd.read_csv("../creditcard.csv")
        # Take a small sample to work with
        sample_df = df.sample(1000, random_state=42)
        return sample_df
    except:
        # If file doesn't exist yet, create synthetic data
        columns = ['Time', 'Amount', 'Class'] + [f'V{i}' for i in range(1, 29)]
        data = np.random.randn(100, 31)
        data[:, 1] = np.abs(data[:, 1]) * 100  # Make Amount positive and realistic
        data[:, 2] = np.random.choice([0, 1], size=100, p=[0.99, 0.01])  # Class with 1% fraud
        return pd.DataFrame(data, columns=columns)

# Function to check API health
def check_api_health():
    try:
        response = requests.get(f"{API_URL}/health", timeout=3)
        if response.status_code == 200:
            health_data = response.json()
            if health_data.get('status') != 'healthy':
                st.warning(f"API status: {health_data.get('status')} - The model may not be loaded correctly.")
                st.info("You need to run the notebooks to train and save the model first.")
                return False
            return True
        else:
            st.error(f"API health check failed with status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        st.error(f"Cannot connect to API at {API_URL}. Please check if the API is running.")
        return False
    except Exception as e:
        st.error(f"Error checking API health: {str(e)}")
        return False

# Function to make prediction
def predict_transaction(transaction_data):
    try:
        # First check if API is accessible
        try:
            health_check = requests.get(f"{API_URL}/health", timeout=2)
            if health_check.status_code != 200:
                st.error(f"API health check failed. Status code: {health_check.status_code}")
                return None
        except requests.exceptions.ConnectionError:
            st.error(f"Cannot connect to API at {API_URL}. Please ensure the API server is running.")
            return None
        except Exception as e:
            st.error(f"Error checking API health: {e}")
            return None
            
        # Now try to make the prediction
        response = requests.post(
            f"{API_URL}/predict",
            json=transaction_data,
            timeout=5
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API returned error status: {response.status_code}")
            st.info(f"Response content: {response.text[:500]}")
            return None
            
    except requests.exceptions.Timeout:
        st.error(f"Request to API timed out. The server might be overloaded.")
        return None
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return None

# Function to generate random transactions for simulation
def generate_random_transaction():
    return {
        "amount": random.uniform(1, 1000),
        "v1": random.uniform(-5, 5),
        "v2": random.uniform(-5, 5),
        "v3": random.uniform(-5, 5),
        "v4": random.uniform(-5, 5),
        "v5": random.uniform(-5, 5),
        "v6": random.uniform(-5, 5),
        "v7": random.uniform(-5, 5),
        "v8": random.uniform(-5, 5),
        "v9": random.uniform(-5, 5),
        "v10": random.uniform(-5, 5),
        "v11": random.uniform(-5, 5),
        "v12": random.uniform(-5, 5),
        "v13": random.uniform(-5, 5),
        "v14": random.uniform(-5, 5),
        "v15": random.uniform(-5, 5),
        "v16": random.uniform(-5, 5),
        "v17": random.uniform(-5, 5),
        "v18": random.uniform(-5, 5),
        "v19": random.uniform(-5, 5),
        "v20": random.uniform(-5, 5),
        "v21": random.uniform(-5, 5),
        "v22": random.uniform(-5, 5),
        "v23": random.uniform(-5, 5),
        "v24": random.uniform(-5, 5),
        "v25": random.uniform(-5, 5),
        "v26": random.uniform(-5, 5),
        "v27": random.uniform(-5, 5),
        "v28": random.uniform(-5, 5)
    }

# Function to generate a slightly abnormal transaction (more likely to be fraud)
def generate_abnormal_transaction():
    tx = generate_random_transaction()
    # Make some features have extreme values
    tx["v1"] = random.uniform(-15, -10)
    tx["v3"] = random.uniform(10, 15)
    tx["v10"] = random.uniform(-15, -10)
    tx["amount"] = random.uniform(800, 1500)
    return tx

# Dashboard page
def show_dashboard():
    st.markdown('<p class="dashboard-title">Credit Card Fraud Detection Dashboard</p>', unsafe_allow_html=True)
    
    # Check API health
    api_status = check_api_health()
    status_col1, status_col2 = st.columns([1, 5])
    
    with status_col1:
        if api_status:
            st.success("API Online")
        else:
            st.error("API Offline")
    
    # Load data
    df = load_sample_data()
    
    # Main stats
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.metric("Total Transactions", f"{len(df):,}")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        fraud_count = df['Class'].sum()
        st.metric("Fraud Detected", f"{int(fraud_count):,}")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        fraud_pct = df['Class'].mean() * 100
        st.metric("Fraud Rate", f"{fraud_pct:.2f}%")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col4:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        avg_amount = df['Amount'].mean()
        st.metric("Avg. Transaction", f"${avg_amount:.2f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Charts
    st.markdown('<p class="section-title">Transaction Analysis</p>', unsafe_allow_html=True)
    
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Transaction Amount Distribution")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(data=df, x='Amount', hue='Class', bins=50, ax=ax, kde=True, palette=['blue', 'red'])
        ax.set_title('Transaction Amount by Class')
        ax.set_xlabel('Amount')
        ax.set_xlim(0, 500)  # Limit x-axis for better visibility
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with chart_col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Fraud Detection Rate")
        
        # Create some dummy time-based data
        today = datetime.now()
        dates = pd.date_range(end=today, periods=30, freq='D')
        fraud_rates = np.random.uniform(0.1, 0.5, size=30)
        detection_rates = np.random.uniform(0.8, 0.99, size=30)
        
        rate_df = pd.DataFrame({
            'Date': dates,
            'Fraud Rate (%)': fraud_rates,
            'Detection Rate (%)': detection_rates
        })
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(rate_df['Date'], rate_df['Fraud Rate (%)'] * 100, 'r-', label='Fraud Rate')
        ax.plot(rate_df['Date'], rate_df['Detection Rate (%)'] * 100, 'g-', label='Detection Rate')
        ax.set_xlabel('Date')
        ax.set_ylabel('Rate (%)')
        ax.set_title('Fraud and Detection Rates Over Time')
        ax.legend()
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Recent transactions table
    st.markdown('<p class="section-title">Recent Transactions</p>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    
    # Create some dummy recent transactions
    n_transactions = 10
    recent_tx = []
    
    for i in range(n_transactions):
        is_fraud = random.random() < 0.2  # 20% chance of being fraud for demonstration
        recent_tx.append({
            'Timestamp': (datetime.now().replace(microsecond=0) - pd.Timedelta(minutes=i)).isoformat(),
            'Transaction ID': f"TX{random.randint(10000, 99999)}",
            'Amount': f"${random.uniform(10, 1000):.2f}",
            'Risk Level': "High" if is_fraud else "Low",
            'Status': "Flagged" if is_fraud else "Approved"
        })
    
    recent_df = pd.DataFrame(recent_tx)
    
    # Apply styling to the table
    def highlight_fraud(val):
        if val == "Flagged":
            return 'background-color: #FFEEEE; color: #FF0000; font-weight: bold'
        elif val == "High":
            return 'background-color: #FFEEEE; color: #FF0000; font-weight: bold'
        elif val == "Approved":
            return 'background-color: #EEFFEE; color: #008800; font-weight: bold'
        elif val == "Low":
            return 'background-color: #EEFFEE; color: #008800; font-weight: bold'
        else:
            return ''
    
    st.dataframe(recent_df.style.applymap(highlight_fraud, subset=['Status', 'Risk Level']), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Transaction simulation page
def simulate_transactions():
    st.markdown('<p class="dashboard-title">Transaction Simulation</p>', unsafe_allow_html=True)
    st.write("Use this page to simulate transactions and test the fraud detection model")
    
    # Check if API is online
    api_status = check_api_health()
    if not api_status:
        st.error("API is offline. Cannot make predictions.")
        st.info("You may need to run the model training notebook first to create the model file.")
        return
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Transaction Details")
        
        # Let user choose between normal, random, or abnormal transaction
        tx_type = st.radio("Transaction Type", ["Custom", "Random Normal", "Suspicious"])
        
        if tx_type == "Custom":
            amount = st.number_input("Amount ($)", min_value=1.0, max_value=5000.0, value=100.0, step=10.0)
            v1 = st.slider("V1", -10.0, 10.0, 0.0, 0.1)
            v3 = st.slider("V3", -10.0, 10.0, 0.0, 0.1)
            v4 = st.slider("V4", -10.0, 10.0, 0.0, 0.1)
            v10 = st.slider("V10", -10.0, 10.0, 0.0, 0.1)
            
            transaction = {
                "amount": amount,
                "v1": v1, "v2": 0.0, "v3": v3, "v4": v4, "v5": 0.0,
                "v6": 0.0, "v7": 0.0, "v8": 0.0, "v9": 0.0, "v10": v10,
                "v11": 0.0, "v12": 0.0, "v13": 0.0, "v14": 0.0, 
                "v15": 0.0, "v16": 0.0, "v17": 0.0, "v18": 0.0, "v19": 0.0,
                "v20": 0.0, "v21": 0.0, "v22": 0.0, "v23": 0.0, "v24": 0.0,
                "v25": 0.0, "v26": 0.0, "v27": 0.0, "v28": 0.0
            }
        elif tx_type == "Random Normal":
            transaction = generate_random_transaction()
            st.write(f"Amount: ${transaction['amount']:.2f}")
        else:  # Suspicious
            transaction = generate_abnormal_transaction()
            st.write(f"Amount: ${transaction['amount']:.2f}")
        
        if st.button("Process Transaction"):
            with st.spinner("Processing..."):
                result = predict_transaction(transaction)
                if result:
                    st.session_state.last_prediction = result
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Prediction Result")
        
        if 'last_prediction' in st.session_state:
            result = st.session_state.last_prediction
            
            # Check if result exists and has the expected keys
            if result is None:
                st.error("No prediction result received from the API.")
                st.info("This could be because the model is not properly loaded or the API is not functioning correctly.")
                return
                
            # Check if result has all the expected keys
            expected_keys = ['is_fraud', 'fraud_probability', 'risk_level', 'suggested_action', 'transaction_id', 'timestamp']
            missing_keys = [key for key in expected_keys if key not in result]
            
            if missing_keys:
                st.error("Invalid response from API: Missing required fields.")
                st.info(f"Missing fields: {', '.join(missing_keys)}")
                st.info("You may need to run the model training notebook first to create the model file.")
                st.write("Received data:", result)
                return
                
            # Display result with styling based on prediction
            is_fraud = result['is_fraud']
            prob = result['fraud_probability']
            risk = result['risk_level']
            action = result['suggested_action']
            
            # Style based on risk
            if risk == "High":
                st.markdown(f"<h2 style='color: #FF4B4B;'>HIGH RISK</h2>", unsafe_allow_html=True)
            elif risk == "Medium":
                st.markdown(f"<h2 style='color: #FFA500;'>MEDIUM RISK</h2>", unsafe_allow_html=True)
            else:
                st.markdown(f"<h2 style='color: #00CC96;'>LOW RISK</h2>", unsafe_allow_html=True)
            
            # Transaction details
            st.markdown(f"**Transaction ID:** {result['transaction_id']}")
            st.markdown(f"**Timestamp:** {result['timestamp']}")
            st.markdown(f"**Fraud Probability:** {prob:.2%}")
            st.markdown(f"**Suggested Action:** {action}")
            
            # Visualization of probability
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.barh(['Fraud Probability'], [prob], color='#FF4B4B' if is_fraud else '#00CC96', alpha=0.7)
            ax.barh(['Fraud Probability'], [1-prob], left=[prob], color='#DDDDDD', alpha=0.3)
            ax.set_xlim(0, 1)
            ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
            ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
            ax.set_xlabel('Probability')
            ax.axvline(x=0.7, color='red', linestyle='--', alpha=0.5)
            ax.axvline(x=0.3, color='orange', linestyle='--', alpha=0.5)
            st.pyplot(fig)
        else:
            st.info("Submit a transaction to see prediction results")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Batch simulation option
    st.markdown('<p class="section-title">Batch Simulation</p>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    
    col3, col4, col5 = st.columns(3)
    
    with col3:
        num_transactions = st.number_input("Number of Transactions", 1, 100, 10)
    
    with col4:
        fraud_ratio = st.slider("Injected Fraud Ratio", 0.0, 1.0, 0.2, 0.05)
    
    with col5:
        simulation_speed = st.selectbox("Simulation Speed", ["Slow", "Medium", "Fast"], 1)
    
    if st.button("Run Batch Simulation"):
        progress_bar = st.progress(0)
        
        # Create a container for showing live updates
        live_updates = st.empty()
        
        # Create a DataFrame to store results
        results_df = pd.DataFrame(columns=[
            'Transaction ID', 'Timestamp', 'Amount', 'Is Fraud', 
            'Fraud Probability', 'Risk Level', 'Action'
        ])
        
        # Set delay based on speed selection
        delay = 1.0 if simulation_speed == "Slow" else 0.5 if simulation_speed == "Medium" else 0.1
        
        # Run simulation
        for i in range(num_transactions):
            # Decide if this transaction should be fraudulent based on the ratio
            should_be_fraud = random.random() < fraud_ratio
            
            # Generate transaction
            if should_be_fraud:
                transaction = generate_abnormal_transaction()
            else:
                transaction = generate_random_transaction()
            
            # Make prediction
            result = predict_transaction(transaction)
            
            if result:
                # Add to results DataFrame
                new_row = pd.DataFrame([{
                    'Transaction ID': result['transaction_id'],
                    'Timestamp': result['timestamp'],
                    'Amount': f"${transaction['amount']:.2f}",
                    'Is Fraud': result['is_fraud'],
                    'Fraud Probability': f"{result['fraud_probability']:.2%}",
                    'Risk Level': result['risk_level'],
                    'Action': result['suggested_action']
                }])
                
                results_df = pd.concat([new_row, results_df.loc[:]], ignore_index=True)
                
                # Update progress and display current status
                progress = (i + 1) / num_transactions
                progress_bar.progress(progress)
                
                # Format the DataFrame for display
                def highlight_risk(val):
                    if val == "High":
                        return 'background-color: #FFDDDD; color: #CC0000'
                    elif val == "Medium":
                        return 'background-color: #FFEECC; color: #FF6600'
                    elif val == "Low":
                        return 'background-color: #DDFFDD; color: #006600'
                    elif val == True:
                        return 'background-color: #FFDDDD; color: #CC0000'
                    elif val == False:
                        return 'background-color: #DDFFDD; color: #006600'
                    else:
                        return ''
                
                # Display the top 5 rows with styling
                display_df = results_df.head(5).copy()
                display_df_styled = display_df.style.applymap(
                    highlight_risk, 
                    subset=['Is Fraud', 'Risk Level']
                )
                
                # Show stats
                fraud_detected = results_df['Is Fraud'].sum()
                with live_updates.container():
                    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                    metrics_col1.metric("Transactions Processed", f"{i+1}/{num_transactions}")
                    metrics_col2.metric("Fraud Detected", f"{fraud_detected}/{i+1} ({fraud_detected/(i+1):.1%})")
                    metrics_col3.metric("Accuracy", f"{((should_be_fraud == result['is_fraud']) * 100):.1f}%")
                    
                    st.dataframe(display_df_styled, use_container_width=True)
            
            time.sleep(delay)
        
        # Show final results
        st.success(f"Simulation completed: {num_transactions} transactions processed")
        
        # Allow downloading the results
        csv_buffer = StringIO()
        results_df.to_csv(csv_buffer, index=False)
        csv_str = csv_buffer.getvalue()
        
        st.download_button(
            label="Download Simulation Results",
            data=csv_str,
            file_name="fraud_simulation_results.csv",
            mime="text/csv"
        )
    
    st.markdown('</div>', unsafe_allow_html=True)

# About page
def show_about():
    st.markdown('<p class="dashboard-title">About This Project</p>', unsafe_allow_html=True)
    
    st.markdown("""
    ## Credit Card Fraud Detection System
    
    This project demonstrates a machine learning-based credit card fraud detection system. It includes:
    
    - **Machine Learning Model**: Trained on the [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) from Kaggle
    - **Real-time API**: FastAPI backend for making predictions on new transactions
    - **Analytics Dashboard**: Monitoring and visualization of fraud patterns
    - **Simulation Tool**: Test the system with various transaction scenarios
    
    ### How It Works
    
    The system analyzes transaction features to identify potentially fraudulent activities. The model is trained on historical data of both legitimate and fraudulent transactions. When a new transaction occurs, it's evaluated in real-time and flagged if suspicious.
    
    ### Key Features
    
    - Real-time fraud detection with probability scores
    - Risk categorization (Low, Medium, High)
    - Suggested actions for fraud analysts
    - Visualization of transaction patterns
    - Batch simulation for system testing
    
    ### Technologies Used
    
    - **Python**: Core programming language
    - **Scikit-learn & XGBoost**: Machine learning libraries
    - **FastAPI**: Backend API framework
    - **Streamlit**: Interactive dashboard
    - **Pandas & NumPy**: Data processing
    - **Matplotlib & Seaborn**: Data visualization
    """)

# Run the selected page
if page == "Dashboard":
    show_dashboard()
elif page == "Simulate Transactions":
    simulate_transactions()
else:
    show_about()
