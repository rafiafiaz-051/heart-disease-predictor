import streamlit as st
import numpy as np
import joblib

model = joblib.load('champion_model.pkl')
scaler = joblib.load('scaler.pkl')

N_FEATURES = scaler.n_features_in_

def predict(age, sex, cp, thalach, ca, oldpeak, thal, chol, trestbps, exang, slope, fbs, restecg):

    features = [
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
    ]

    # optional engineered features (ONLY if model expects 15 features)
    if N_FEATURES == 15:

        age_group = 0
        if age < 40:
            age_group = 0
        elif age < 55:
            age_group = 1
        elif age < 70:
            age_group = 2
        else:
            age_group = 3

        chol_high = 1 if chol > 240 else 0

        features.append(age_group)
        features.append(chol_high)

    input_data = np.array([features])
    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    return int(prediction), round(float(probability) * 100, 1)


st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="centered"
)

st.title("Heart Disease Risk Predictor")
st.write("Fill in the fields below and click Predict to assess risk.")
st.divider()

col1, col2 = st.columns(2)

with col1:

    age = st.number_input("Age (years)", 20, 80, 50)
    sex = st.selectbox("Gender", [1, 0], format_func=lambda x: "Male" if x == 1 else "Female")

    cp = st.selectbox(
        "Chest Pain Type",
        [0, 1, 2, 3],
        format_func=lambda x: {
            0: "Typical Angina",
            1: "Atypical Angina",
            2: "Non-Anginal Pain",
            3: "No Pain"
        }[x]
    )

    thalach = st.number_input("Max Heart Rate", 70, 210, 150)
    trestbps = st.number_input("Resting BP", 80, 250, 120)

    fbs = st.selectbox("Fasting Blood Sugar >120", [0, 1])
    restecg = st.selectbox("Rest ECG", [0, 1, 2])

with col2:

    exang = st.selectbox("Exercise Angina", [0, 1])
    ca = st.selectbox("Major Vessels", [0, 1, 2, 3])

    oldpeak = st.number_input("Oldpeak", 0.0, 6.5, 1.0)
    slope = st.selectbox("Slope", [0, 1, 2])
    thal = st.selectbox("Thal", [1, 2, 3])

    chol = st.number_input("Cholesterol", 100, 600, 240)

st.divider()

if st.button("Predict", use_container_width=True):

    try:

        prediction, probability = predict(
            age, sex, cp, thalach, ca,
            oldpeak, thal, chol, trestbps,
            exang, slope, fbs, restecg
        )

        st.subheader("Result")

        if prediction == 1:
            st.error("Heart Disease Detected")
        else:
            st.success("No Heart Disease Detected")

        st.metric("Risk Probability", f"{probability}%")

    except Exception as e:
        st.error(f"Prediction error: {e}")
