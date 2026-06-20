import streamlit as st
import numpy as np
import joblib

# Load model + scaler
model = joblib.load("champion_model.pkl")
scaler = joblib.load("scaler.pkl")

# Page config
st.set_page_config(page_title="Heart Disease Predictor", page_icon="💓", layout="centered")

st.title("💓 Heart Disease Prediction App")
st.write("Enter basic details below to predict risk")

st.markdown("---")

# -------------------------
# SIMPLE USER INPUTS
# -------------------------
age = st.slider("Age", 20, 100, 40)

sex = st.selectbox("Gender", ["Male", "Female"])
sex = 1 if sex == "Male" else 0

cp = st.selectbox("Chest Pain Type", [
    "Typical Angina",
    "Atypical Angina",
    "Non-anginal",
    "Asymptomatic"
])
cp = ["Typical Angina","Atypical Angina","Non-anginal","Asymptomatic"].index(cp)

trestbps = st.slider("Resting Blood Pressure", 80, 200, 120)

chol = st.slider("Cholesterol", 100, 600, 200)

thalach = st.slider("Max Heart Rate", 60, 220, 150)

exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
exang = 1 if exang == "Yes" else 0

st.markdown("---")

# -------------------------
# DEFAULT (missing features)
# -------------------------
fbs = 0
restecg = 1
oldpeak = 1.0
slope = 1
ca = 0
thal = 2

# -------------------------
# PREDICTION
# -------------------------
if st.button("🔍 Predict Risk"):

    # ⚠️ MUST MATCH TRAINING ORDER (13 FEATURES)
    input_data = np.array([[
        age,
        sex,
        cp,
        trestbps,
        chol,
        fbs,
        restecg,
        thalach,
        exang,
        oldpeak,
        slope,
        ca,
        thal
    ]])

    # scale
    input_scaled = scaler.transform(input_data)

    # predict
    prediction = model.predict(input_scaled)[0]

    # output
    if prediction == 1:
        st.error("⚠ High Risk of Heart Disease")
        st.write("Please consult a doctor.")
    else:
        st.success("✅ Low Risk of Heart Disease")
        st.write("No major risk detected. Stay healthy ❤️")
