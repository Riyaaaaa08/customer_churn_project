import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Set up page header
st.title("📊 Telco Customer Churn Predictor App")
st.write("Enter a customer's contract details below to calculate their churn risk in real-time.")
st.write("---")

# 1. Load our saved model brain and scaler back into memory
with open("churn_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

# 2. Collect User Inputs
st.header("👤 Customer Profile Input")

# Numeric fields
tenure = st.slider("Customer Tenure (Months with company):", min_value=1, max_value=72, value=12)
monthly_charges = st.number_input("Monthly Charges ($):", min_value=10.0, max_value=150.0, value=65.0)

# Categorical fields
contract = st.selectbox("Contract Type:", ["Month-to-month", "One year", "Two year"])
internet = st.selectbox("Internet Service Type:", ["Fiber optic", "DSL", "No"])
security = st.selectbox("Online Security Service:", ["No", "Yes"])
support = st.selectbox("Tech Support Service:", ["No", "Yes"])
billing = st.selectbox("Paperless Billing:", ["Yes", "No"])

# 3. Build the exact matching 30-column DataFrame row
# We initialize all columns to 0 or baseline averages to prevent math distortion
data_dict = {
    'SeniorCitizen': [0],
    'tenure': [tenure],
    'MonthlyCharges': [monthly_charges],
    'TotalCharges': [tenure * monthly_charges], # Approximation of total charges
    'gender_Male': [0],
    'Partner_Yes': [0],
    'Dependents_Yes': [0],
    'PhoneService_Yes': [1], # Assume default baseline phone service
    'MultipleLines_No phone service': [0],
    'MultipleLines_Yes': [0],
    'InternetService_Fiber optic': [1 if internet == "Fiber optic" else 0],
    'InternetService_No': [1 if internet == "No" else 0],
    'OnlineSecurity_No internet service': [1 if internet == "No" else 0],
    'OnlineSecurity_Yes': [1 if security == "Yes" else 0],
    'OnlineBackup_No internet service': [1 if internet == "No" else 0],
    'OnlineBackup_Yes': [0],
    'DeviceProtection_No internet service': [1 if internet == "No" else 0],
    'DeviceProtection_Yes': [0],
    'TechSupport_No internet service': [1 if internet == "No" else 0],
    'TechSupport_Yes': [1 if support == "Yes" else 0],
    'StreamingTV_No internet service': [1 if internet == "No" else 0],
    'StreamingTV_Yes': [0],
    'StreamingMovies_No internet service': [1 if internet == "No" else 0],
    'StreamingMovies_Yes': [0],
    'Contract_One year': [1 if contract == "One year" else 0],
    'Contract_Two year': [1 if contract == "Two year" else 0],
    'PaperlessBilling_Yes': [1 if billing == "Yes" else 0],
    'PaymentMethod_Credit card (automatic)': [1], # Assume baseline stable payment method
    'PaymentMethod_Electronic check': [0],
    'PaymentMethod_Mailed check': [0]
}

# Convert dictionary to DataFrame to lock the columns in the precise order
final_df = pd.DataFrame(data_dict)

# Scale inputs using the exact same scaler rules
input_scaled = scaler.transform(final_df.values)

# 4. Trigger prediction engine on button click
if st.button("🔮 Calculate Churn Risk"):
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]
    
    st.write("---")
    if prediction == 1:
        st.error(f"⚠️ **High Churn Risk!** This customer is highly likely to leave. (Risk Score: {probability:.1%})")
    else:
        st.success(f"✅ **Low Risk.** This customer appears stable and loyal. (Risk Score: {probability:.1%})")