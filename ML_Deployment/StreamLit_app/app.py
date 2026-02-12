import streamlit as st
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Utils.model_loader import model

# title
st.title("Insurance Cost PRediction app")
st.write("Enter the health details to predict premium price")

# input fields
age = st.number_input("Age", min_value=1, max_value=100, value=30)
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
diabetes = st.selectbox("Diabetes", [0, 1])
bp = st.selectbox("Blood Pressure Problems", [0, 1])
transplants = st.selectbox("Any Transplants", [0, 1])
chronic = st.selectbox("Any Chronic Diseases", [0, 1])
allergies = st.selectbox("Known Allergies", [0, 1])
cancer = st.selectbox("History of Cancer in Family", [0, 1])
surgeries = st.selectbox("Number of Major Surgeries", [0, 1, 2, 3])

# Predict button
if st.button("PRedict premium price"):
    features = np.array([[age, diabetes, bp, transplants,
        chronic, allergies, cancer,
        surgeries, bmi]])
    prediction = model.predict(features)[0]
    st.success(f'Estimated Premium Price cost :{round(prediction, 2)}')

