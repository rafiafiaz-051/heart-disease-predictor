import streamlit as st
import pandas as pd
import joblib
import numpy as np

model = joblib.load("champion_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Heart Disease Prediction System")

age = st.number_input("Age", 20, 100, 50)
sex = st.selectbox("Sex", [0,1])
cp = st.selectbox("Chest Pain Type", [0,1,2,3])
trestbps = st.number_input("Resting Blood Pressure", 80, 250, 120)
chol = st.number_input("Cholesterol", 100, 600, 200)
fbs = st.selectbox("Fasting Blood Sugar", [0,1])
restecg = st.selectbox("Rest ECG", [0,1,2])
thalach = st.number_input("Maximum Heart Rate", 60, 250, 150)
exang = st.selectbox("Exercise Induced Angina", [0,1])
oldpeak = st.number_input("Oldpeak", 0.0, 10.0, 1.0)
slope = st.selectbox("Slope", [0,1,2])
ca = st.selectbox("CA", [0,1,2,3,4])
thal = st.selectbox("Thal", [0,1,2,3])

age_group_encoded = 0

if age < 40:
    age_group_encoded = 0
elif age < 55:
    age_group_encoded = 1
elif age < 65:
    age_group_encoded = 2
else:
    age_group_encoded = 3

chol_high = 1 if chol > 240 else 0

if st.button("Predict"):

    data = pd.DataFrame([[
        age, sex, cp, trestbps, chol,
        fbs, restecg, thalach, exang,
        oldpeak, slope, ca, thal,
        age_group_encoded, chol_high
    ]], columns=[
        'age','sex','cp','trestbps','chol',
        'fbs','restecg','thalach','exang',
        'oldpeak','slope','ca','thal',
        'age_group_encoded','chol_high'
    ])

    data_scaled = scaler.transform(data)

    prediction = model.predict(data_scaled)[0]

    if prediction == 1:
        st.error("High Risk of Heart Disease")
    else:
        st.success("Low Risk of Heart Disease")