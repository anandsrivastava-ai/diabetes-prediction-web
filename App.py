import streamlit as st
import joblib
import numpy as np

# Load your trained model
lr = joblib.load("diabetes_model.pkl")

# App title
st.title("🩺 Diabetes Prediction App")

st.write("Enter details below to check diabetes risk:")

# Input fields
glucose = st.number_input("Glucose Level", min_value=0, max_value=300, value=120)
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
age = st.number_input("Age", min_value=1, max_value=120, value=33)

# Prediction
if st.button("Predict"):
    # Arrange inputs into an array
    input_data = np.array([[glucose, bmi, age]])
    
    prediction = lr.predict(input_data)[0]

    if prediction == 1:
        st.error("⚠️ The model predicts that the patient **has a risk of diabetes.**")
    else:
        st.success("✅ The model predicts that the patient **is not likely to have diabetes.**")
        