import streamlit as st
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load your dataset (replace this with your actual file path)
@st.cache
def load_data():
    filepath = './Big_Black_Money_Dataset.csv'  # Update with actual path
    df = pd.read_csv(filepath)
    return df

# Function to train Random Forest model
def train_model(df):
    # Select relevant features
    X = df[['amount_usd', 'transaction_type', 'source_of_money', 'country', 'destination_country']]
    
    # Convert categorical variables to numerical (one-hot encoding)
    X = pd.get_dummies(X, drop_first=True)
    
    # Define target variable
    y = df['risk_score'].apply(lambda x: 1 if x >= 8 else 0)  # 1 for high-risk, 0 for low-risk
    
    # Train the Random Forest model
    rf = RandomForestClassifier(n_estimators=100, random_state=1)
    rf.fit(X, y)
    
    return rf, X.columns  # Return model and feature names

# Function to make predictions
def predict_risk(model, input_data):
    prediction = model.predict(input_data)
    return prediction

# Streamlit app
st.title("High-Risk Transaction Prediction App")

# Load the dataset
df = load_data()

# Train the model
model, feature_columns = train_model(df)

# User inputs for the features
st.header("Input Features for Transaction")

amount_usd = st.number_input("Transaction Amount (USD)", min_value=1000, max_value=5000000, value=10000)

# Categorical inputs (encoded as strings initially)
transaction_type = st.selectbox("Transaction Type", ["Offshore Transfer", "Stocks Transfer", "Cash Withdrawal", "Cryptocurrency", "Property Purchase"])
source_of_money = st.selectbox("Source of Money", ["Legal", "Illegal"])
country = st.selectbox("Origin Country", df['country'].unique())
destination_country = st.selectbox("Destination Country", df['destination_country'].unique())

# Prepare the input data for prediction
input_data = pd.DataFrame([[amount_usd, transaction_type, source_of_money, country, destination_country]], 
                          columns=['amount_usd', 'transaction_type', 'source_of_money', 'country', 'destination_country'])

# Encode the input data using the same encoding as the training data
input_data_encoded = pd.get_dummies(input_data)
input_data_encoded = input_data_encoded.reindex(columns=feature_columns, fill_value=0)  # Ensure same feature columns

# Predict high-risk when the button is clicked
if st.button("Predict Risk"):
    prediction = predict_risk(model, input_data_encoded)
    if prediction[0] == 1:
        st.error("The transaction is likely high-risk.")
    else:
        st.success("The transaction is likely low-risk.")
