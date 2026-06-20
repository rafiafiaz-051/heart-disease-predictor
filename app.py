import streamlit as st
import numpy as np
import joblib

model = joblib.load('champion_model.pkl')
scaler = joblib.load('scaler.pkl')

N_FEATURES = scaler.n_features_in_

HIDDEN_MEANS = {
    'trestbps': 131.6,
    'chol': 246.5,
    'fbs': 0.15,
    'restecg': 0.53,
    'exang': 0.33,
    'slope': 1.40,
    'age_group_encoded': 1.8,
    'chol_high': 0.37,
}

def predict(age, sex, cp, thalach, ca, oldpeak, thal, chol):

    features_13 = [
        age,
        sex,
        cp,
        HIDDEN_MEANS['trestbps'],
        chol,
        HIDDEN_MEANS['fbs'],
        HIDDEN_MEANS['restecg'],
        thalach,
        HIDDEN_MEANS['exang'],
        oldpeak,
        HIDDEN_MEANS['slope'],
        ca,
        thal
    ]

    chol_high = 1 if chol > 240 else 0

    if N_FEATURES == 15:

        if age < 40:
            age_group = 0
        elif age < 55:
            age_group = 1
        elif age < 70:
            age_group = 2
        else:
            age_group = 3

        chol_high = 1 if HIDDEN_MEANS['chol'] > 240 else 0

        features_13.append(age_group)
        features_13.append(chol_high)

    input_data = np.array([features_13])

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

st.write(
    "Fill in the fields below and click Predict to assess risk."
)

st.divider()

col1, col2 = st.columns(2)

with col1:

    age = st.number_input(
        "Age (years)",
        min_value=20,
        max_value=80,
        value=50,
        step=1
    )

    sex = st.selectbox(
        "Gender",
        options=[1, 0],
        format_func=lambda x: "Male" if x == 1 else "Female"
    )

    cp = st.selectbox(
    "Chest Pain Type",
    options=[0, 1, 2, 3],
    format_func=lambda x: {
        0: "Typical Angina (Chest pain during physical activity)",
        1: "Atypical Angina (Unusual chest pain that may occur at any time)",
        2: "Non-Anginal Chest Pain (Chest pain not related to the heart)",
        3: "No Chest Pain Symptoms"
    }[x]
)

    thalach = st.number_input(
        "Max Heart Rate (bpm)",
        min_value=70,
        max_value=210,
        value=150,
        step=1
    )

with col2:

    ca = st.selectbox(
    "Number of Major Blood Vessels Visible in Test",
    options=[0, 1, 2, 3],
    help="Usually taken from an angiography test. Select the number reported by the doctor."
)

    oldpeak = st.number_input(
    "ST Depression (Oldpeak)",
    min_value=0.0,
    max_value=6.5,
    value=1.0,
    step=0.1,
    format="%.1f",
    help="Value taken from a cardiac stress test report. Do not guess this value."
)

    thal = st.selectbox(
    "Thalassemia Test Result",
    options=[1, 2, 3],
    format_func=lambda x: {
        1: "Normal",
        2: "Fixed Defect",
        3: "Reversible Defect"
    }[x],
    help="Use the value from your medical report."
)
    
    chol = st.number_input(
    "Cholesterol (mg/dl)",
    min_value=100,
    max_value=600,
    value=240,
    step=1,
    help="Blood cholesterol level from lab report"
)

st.divider()

if st.button("Predict", use_container_width=True, type="primary"):

    try:

        prediction, probability = predict(
            age,
            sex,
            cp,
            thalach,
            ca,
            oldpeak,
            thal,
            chol
        )

        st.subheader("Result")

        if prediction == 1:

            st.error("Heart Disease Detected")

            st.metric(
                "Risk Probability",
                f"{probability}%"
            )

            st.warning(
                "This is a screening tool only. Please consult a cardiologist."
            )

        else:

            st.success("No Heart Disease Detected")

            st.metric(
                "Risk Probability",
                f"{probability}%"
            )

            st.info(
                "Low risk detected. Maintain a healthy lifestyle and schedule regular check-ups."
            )

    except Exception as e:

        st.error(f"Prediction error: {e}")

        st.info(
            "Please verify that the saved model and scaler files match."
        )



