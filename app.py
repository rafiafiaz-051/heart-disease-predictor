import streamlit as st
import numpy as np
import joblib

# Load model & scaler
model  = joblib.load('champion_model.pkl')
scaler = joblib.load('scaler.pkl')

# -------------------------
# AUTO FILLED VALUES
# -------------------------
HIDDEN_MEANS = {
    'trestbps': 131.6,
    'chol'    : 246.5,
    'fbs'     : 0.15,
    'restecg' : 0.53,
    'exang'   : 0.33,
    'slope'   : 1.40,
}

# -------------------------
# PREDICTION FUNCTION
# -------------------------
def predict(age, sex, cp, thalach, oldpeak, ca, thal):

    input_data = np.array([[
        age,                         # 1
        sex,                         # 2
        cp,                          # 3
        HIDDEN_MEANS['trestbps'],    # 4
        HIDDEN_MEANS['chol'],        # 5
        HIDDEN_MEANS['fbs'],         # 6
        HIDDEN_MEANS['restecg'],     # 7
        thalach,                     # 8
        HIDDEN_MEANS['exang'],       # 9
        oldpeak,                     # 10
        HIDDEN_MEANS['slope'],       # 11
        ca,                          # 12
        thal                         # 13
    ]])

    scaled = scaler.transform(input_data)
    pred = model.predict(scaled)[0]
    prob = model.predict_proba(scaled)[0][1]

    return pred, round(prob * 100, 1)

# -------------------------
# UI CONFIG
# -------------------------
st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️")

st.title("❤️ Heart Disease Predictor")

st.markdown("Enter key health details:")

# -------------------------
# CLEAN UI (7 FEATURES ONLY)
# -------------------------
age = st.number_input("Age", 20, 80, 50)

sex = st.selectbox("Gender", [1, 0], format_func=lambda x: "Male" if x == 1 else "Female")

cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])

thalach = st.number_input("Max Heart Rate", 70, 210, 150)

oldpeak = st.number_input("ST Depression", 0.0, 6.5, 1.0)

ca = st.selectbox("Major Vessels (0–3)", [0, 1, 2, 3])

thal = st.selectbox("Thalassemia", [1, 2, 3])

# -------------------------
# PREDICTION
# -------------------------
if st.button("Predict"):
    pred, prob = predict(age, sex, cp, thalach, oldpeak, ca, thal)

    if pred == 1:
        st.error("High Risk Detected")
        st.metric("Risk %", f"{prob}%")
    else:
        st.success("Low Risk Detected")
        st.metric("Risk %", f"{prob}%")

