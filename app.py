import streamlit as st
import numpy as np
import joblib

# Load model & scaler
model = joblib.load("champion_model.pkl")
scaler = joblib.load("scaler.pkl")

# ---------------------------
# DEFAULT VALUES (from dataset approx)
# ---------------------------
defaults = {
    "fbs": 0,
    "restecg": 1,
    "oldpeak": 1.0,
    "slope": 1,
    "ca": 0,
    "thal": 2
}

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(page_title="Heart Risk App", page_icon="💓", layout="centered")

st.markdown("""
    <style>
    .main { background-color: #f5f7ff; }
    h1 { color: #c9184a; text-align:center; }
    </style>
""", unsafe_allow_html=True)

st.title("💓 Heart Disease Risk Predictor")
st.write("Enter basic patient details to predict risk")

st.markdown("---")

# ---------------------------
# SIMPLE USER INPUTS
# ---------------------------
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

# ---------------------------
# PREDICTION
# ---------------------------
if st.button("🔍 Predict Risk"):

    # FULL 13-FEATURE INPUT (IMPORTANT ORDER MUST MATCH MODEL)
    input_data = np.array([[
        age,
        sex,
        cp,
        trestbps,
        chol,
        defaults["fbs"],
        defaults["restecg"],
        thalach,
        exang,
        defaults["oldpeak"],
        defaults["slope"],
        defaults["ca"],
        defaults["thal"]
    ]])

    # scale input
    input_scaled = scaler.transform(input_data)

    # prediction
    prediction = model.predict(input_scaled)[0]

    # output
    if prediction == 1:
        st.error("⚠ High Risk of Heart Disease")
        st.write("Consult a doctor for further evaluation.")
    else:
        st.success("✅ Low Risk of Heart Disease")
        st.write("No major risk detected.")
