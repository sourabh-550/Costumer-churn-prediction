import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model and preprocessor
model = joblib.load("models/xgboost_churn_model.pkl")
preprocessor = joblib.load("models/preprocessor.pkl")

st.set_page_config(page_title="Customer Churn Predictor", layout="centered")

st.title("📉 Customer Churn Prediction System")
st.write("Predict whether a customer is likely to churn based on service usage.")

# -------- User Input --------
st.header("Customer Information")

gender = st.selectbox("Gender", ["Male", "Female"])
senior = st.selectbox("Senior Citizen", ["Yes", "No"])
partner = st.selectbox("Has Partner", ["Yes", "No"])
dependents = st.selectbox("Has Dependents", ["Yes", "No"])

tenure = st.slider("Tenure (months)", 0, 72, 12)

phone_service = st.selectbox("Phone Service", ["Yes", "No"])
internet_service = st.selectbox(
    "Internet Service", ["DSL", "Fiber optic", "No"]
)

contract = st.selectbox(
    "Contract Type",
    ["Month-to-month", "One year", "Two year"]
)

payment_method = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

monthly_charges = st.number_input(
    "Monthly Charges", min_value=0.0, value=70.0
)

total_charges = st.number_input(
    "Total Charges", min_value=0.0, value=1000.0
)

# -------- Prediction --------
if st.button("Predict Churn"):
    input_data = pd.DataFrame({
        "gender": [gender],
        "SeniorCitizen": [1 if senior == "Yes" else 0],
        "Partner": [partner],
        "Dependents": [dependents],
        "tenure": [tenure],
        "PhoneService": [phone_service],
        "InternetService": [internet_service],
        "Contract": [contract],
        "PaymentMethod": [payment_method],
        "MonthlyCharges": [monthly_charges],
        "TotalCharges": [total_charges]
    })

    # Drop customerID if exists (safety)
    if "customerID" in input_data.columns:
        input_data = input_data.drop(columns=["customerID"])

    # Apply preprocessing
    processed_input = preprocessor.transform(input_data)

    # Prediction
    churn_prob = model.predict_proba(processed_input)[0][1]
    churn_pred = "Yes" if churn_prob >= 0.5 else "No"

    st.subheader("Prediction Result")

    st.write(f"**Churn Probability:** {churn_prob:.2f}")

    if churn_pred == "Yes":
        st.error("⚠️ Customer is likely to churn")
    else:
        st.success("✅ Customer is likely to stay")
