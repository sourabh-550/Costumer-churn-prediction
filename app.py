import streamlit as st
import pandas as pd
import joblib
import numpy as np

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="Customer Churn Prediction System",
    page_icon="📉",
    layout="wide"
)

# -------------------------------
# Load Artifacts
# -------------------------------
model = joblib.load("models/xgboost_churn_model.pkl")
preprocessor = joblib.load("models/preprocessor.pkl")

# -------------------------------
# Sidebar - Project Info
# -------------------------------
st.sidebar.title("📌 Project Overview")

st.sidebar.markdown("""
### Customer Churn Prediction System

This project predicts whether a customer is likely to churn using:

- 🔹 XGBoost Classifier  
- 🔹 Hyperparameter tuning (RandomizedSearchCV)  
- 🔹 Class imbalance handling  
- 🔹 Production-grade preprocessing pipeline  

### Model Performance
- ROC-AUC: **0.85**
- Recall: **0.81**
- Precision: **0.51**

The model prioritizes catching churners (high recall), aligning with business goals.
""")

st.sidebar.markdown("---")
st.sidebar.markdown("👨‍💻 Built by:Sourabh Saxena")
st.sidebar.markdown("🎯 Role: ML Engineer Project")

# -------------------------------
# Main Title
# -------------------------------
st.title("📉 Customer Churn Prediction System")
st.markdown("""
This interactive application predicts the likelihood of a telecom customer churning.
It uses a tuned **XGBoost model** trained on historical customer behavior data.
""")

st.markdown("---")

# -------------------------------
# Input Section
# -------------------------------
st.header("📝 Customer Information")

col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", ["Yes", "No"])
    partner = st.selectbox("Has Partner", ["Yes", "No"])
    dependents = st.selectbox("Has Dependents", ["Yes", "No"])

with col2:
    tenure = st.slider("Tenure (Months)", 0, 72, 12)
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

with col3:
    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )
    monthly_charges = st.number_input(
        "Monthly Charges", min_value=0.0, value=70.0
    )
    total_charges = st.number_input(
        "Total Charges", min_value=0.0, value=1000.0
    )

st.markdown("---")

# -------------------------------
# Prediction Logic
# -------------------------------
if st.button("🔮 Predict Churn Probability"):

    input_data = pd.DataFrame({
        "gender": [gender],
        "SeniorCitizen": [1 if senior == "Yes" else 0],
        "Partner": [partner],
        "Dependents": [dependents],
        "tenure": [tenure],
        "PhoneService": ["Yes"],
        "MultipleLines": ["No"],
        "InternetService": [internet_service],
        "OnlineSecurity": ["No"],
        "OnlineBackup": ["No"],
        "DeviceProtection": ["No"],
        "TechSupport": ["No"],
        "StreamingTV": ["No"],
        "StreamingMovies": ["No"],
        "Contract": [contract],
        "PaperlessBilling": ["Yes"],
        "PaymentMethod": [payment_method],
        "MonthlyCharges": [monthly_charges],
        "TotalCharges": [total_charges]
    })

    processed_input = preprocessor.transform(input_data)

    churn_prob = model.predict_proba(processed_input)[0][1]
    churn_pred = "Yes" if churn_prob >= 0.5 else "No"

    st.markdown("## 📊 Prediction Result")

    colA, colB = st.columns(2)

    with colA:
        st.metric("Churn Probability", f"{churn_prob:.2%}")

    with colB:
        if churn_pred == "Yes":
            st.error("⚠️ High Risk of Churn")
        else:
            st.success("✅ Low Risk of Churn")

    st.progress(float(churn_prob))

    # Business Insight
    st.markdown("### 💡 Business Interpretation")

    if churn_prob > 0.7:
        st.warning("Immediate retention action recommended (Offer discounts / Contract upgrade).")
    elif churn_prob > 0.4:
        st.info("Moderate churn risk. Consider engagement campaigns.")
    else:
        st.success("Customer likely to stay. Maintain current engagement.")

    st.markdown("---")

    st.markdown("""
    ### 🔍 Model Insight
    The prediction is based on patterns learned from:
    - Contract type
    - Tenure
    - Monthly charges
    - Service subscriptions
    - Payment method
    
    The model is optimized to minimize missed churners (high recall).
    """)
