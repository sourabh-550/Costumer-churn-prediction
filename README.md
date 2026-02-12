# 📉 Customer Churn Prediction System

An end-to-end, production-ready **Customer Churn Prediction System** built using **XGBoost**, featuring modular architecture, hyperparameter tuning, class imbalance handling, and an interactive Streamlit deployment.

---

## 🚀 Project Overview

Customer churn is a critical problem in subscription-based industries. Acquiring new customers is significantly more expensive than retaining existing ones.

This project builds a complete machine learning pipeline that:

- Predicts whether a customer is likely to churn
- Handles class imbalance using `scale_pos_weight`
- Uses cross-validated hyperparameter tuning (RandomizedSearchCV)
- Optimizes for high recall to minimize missed churners
- Deploys the model via a professional Streamlit application

The system follows industry best practices with a clean separation between experimentation and production code.

---

## 🎯 Business Objective

The goal of this system is to:

- Identify customers at high risk of churn
- Enable proactive retention strategies
- Reduce revenue loss
- Provide probability-based risk scoring for decision-making

The model is intentionally optimized for **high recall**, aligning with real-world business priorities.

---

## 📊 Model Performance

After hyperparameter tuning:

- **ROC-AUC:** 0.85  
- **Recall:** 0.81  
- **Precision:** 0.51  
- **Accuracy:** 0.74  

The model prioritizes identifying churners (high recall), even if it slightly sacrifices precision.

---

## 🧠 Machine Learning Workflow

The project follows a structured ML workflow:

1. **Exploratory Data Analysis (EDA)**
   - Class imbalance analysis
   - Feature behavior visualization
   - Business-driven insights

2. **Preprocessing**
   - Data type correction
   - Missing value imputation (median)
   - One-hot encoding for categorical variables
   - Modular preprocessing pipeline

3. **Baseline Models**
   - Logistic Regression
   - Decision Tree
   - Performance comparison

4. **XGBoost Training**
   - Gradient Boosted Trees
   - Imbalance handling using `scale_pos_weight`

5. **Hyperparameter Tuning**
   - RandomizedSearchCV (5-fold cross-validation)
   - ROC-AUC optimization

6. **Deployment**
   - Streamlit web application
   - Reusable preprocessing + model artifacts
   - Probability-based predictions

---


---

## ⚙️ Tech Stack

- Python
- Pandas & NumPy
- Scikit-learn
- XGBoost
- Streamlit
- Joblib

---


