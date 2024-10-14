import streamlit as st
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load your dataset
@st.cache  # Cache the function to avoid re-running on every interaction
def load_data():
    filepath = './Big_Black_Money_Dataset.csv'
    df = pd.read_csv(filepath)
    df = df.rename(columns={
        'Amount (USD)': 'amount_usd',
        'Transaction Type': 'transaction_type',
        'Destination Country': 'destination_country',
        'Source of Money': 'source_of_money',
        'Money Laundering Risk Score': 'risk_score',
        'Shell Companies Involved': 'shell_companies',
        'Tax Haven Country': 'tax_haven_country'
    })
    return df

# Function to train Random Forest model
def train_model(df):
    # Check if columns exist before dropping them
    columns_to_drop = ['risk_score', 'transaction_id', 'person_involved', 'transaction_date']
    columns_in_data = [col for col in columns_to_drop if col in df.columns]
    
    # Drop target & unnecessary columns
    X = df.drop(columns=columns_in_data)
    y = df['risk_score'].apply(lambda x: 1 if x >= 8 else 0)  # Binary classification (high-risk = 1, low-risk = 0)

    # One-hot encode categorical variables
    X = pd.get_dummies(X, drop_first=True)
    
    # Standardize the numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train the Random Forest model
    rf = RandomForestClassifier(n_estimators=100, random_state=1)
    rf.fit(X_scaled, y)
    
    return rf, scaler, X.columns  # Return model, scaler, and column names for input later

# Function to make predictions
def predict_risk(model, scaler, columns, input_data):
    input_df = pd.DataFrame([input_data], columns=columns)
    
    # Standardize the input data
    input_scaled = scaler.transform(input_df)
    
    prediction = model.predict(input_scaled)
    return prediction

# Streamlit app
st.title("High-Risk Transaction Prediction App")

# Load the dataset
df = load_data()

# Train the Random Forest model
model, scaler, columns = train_model(df)

# User inputs for the features
st.header("Input Features for Transaction")

# Create input fields for features relevant to the transaction
amount_usd = st.number_input("Transaction Amount (USD)", min_value=1000.0, max_value=5000000.0)
transaction_type = st.selectbox("Transaction Type", df['transaction_type'].unique())
destination_country = st.selectbox("Destination Country", df['destination_country'].unique())
source_of_money = st.selectbox("Source of Money", df['source_of_money'].unique())
shell_companies = st.number_input("Number of Shell Companies Involved", min_value=0, max_value=10)
tax_haven_country = st.selectbox("Tax Haven Country", df['tax_haven_country'].unique())

# Prepare the input data as a dictionary
input_data = {
    'amount_usd': amount_usd,
    'transaction_type_' + transaction_type: 1,
    'destination_country_' + destination_country: 1,
    'source_of_money_' + source_of_money: 1,
    'shell_companies': shell_companies,
    'tax_haven_country_' + tax_haven_country: 1
}

# Fill in missing columns with 0 for the categorical variables that are not selected
for col in columns:
    if col not in input_data:
        input_data[col] = 0

# Predict high-risk when the button is clicked
if st.button("Predict High-Risk Transaction"):
    prediction = predict_risk(model, scaler, columns, input_data)
    if prediction[0] == 1:
        st.error("This transaction is likely to be high-risk.")
    else:
        st.success("This transaction is unlikely to be high-risk.")
