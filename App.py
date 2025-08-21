import streamlit as st
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

# Load your trained Logistic Regression model
# Make sure you retrain your model using LogisticRegression() and save as 'diabetes_model.pkl'
lr = joblib.load("diabetes_model.pkl")

# App title
st.title("🩺 Diabetes Prediction App")
st.write("Enter details below to check diabetes risk:")

# Input fields
age = st.number_input("Age", 0, 120, 33, key="input_age_v2")
bmi = st.number_input("BMI", 0.0, 100.0, 25.0, key="input_bmi_v2")
glucose = st.number_input("Glucose Level", 0, 300, 120, key="input_glucose_v2")

# Predict button
if st.button("Predict", key="btn_predict_v2"):
    # Arrange inputs into an array (match training order)
    input_data = np.array([[glucose, bmi, age]])
    
    # Make prediction using Logistic Regression
    prediction = lr.predict(input_data)[0]  # 0 or 1
    prediction_prob = lr.predict_proba(input_data)[0][1]  # probability of diabetes

    # Show result
    if prediction == 1:
        st.error(f"⚠️ The model predicts that the patient **has a risk of diabetes.** (Probability: {prediction_prob:.2f})")
    else:
        st.success(f"✅ The model predicts that the patient **is not likely to have diabetes.** (Probability: {prediction_prob:.2f})")
