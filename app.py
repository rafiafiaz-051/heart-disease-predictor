# ============================================================
#  Heart Disease Predictor — Streamlit App
#  Just run:  streamlit run app.py
# ============================================================

import streamlit as st
import numpy as np
import joblib

# ── 1. Load model & scaler (must be in same folder) ─────────
model  = joblib.load('champion_model.pkl')
scaler = joblib.load('scaler.pkl')

# ── 2. Mean values for the 6 hidden fields ──────────────────
#    These were calculated from your training dataset
HIDDEN_MEANS = {
    'trestbps': 131.6,   # Resting blood pressure
    'chol'    : 246.5,   # Cholesterol
    'fbs'     : 0.15,    # Fasting blood sugar
    'restecg' : 0.53,    # Resting ECG
    'exang'   : 0.33,    # Exercise induced angina
    'slope'   : 1.40,    # Slope of ST segment
}

# ── 3. PREDICTION FUNCTION ───────────────────────────────────
#    This is the function that FIXES the shape error
def predict(age, sex, cp, thalach, ca, oldpeak, thal):

    # Build full 13-feature input in EXACT training order
    input_data = np.array([[
        age,                        # 1. age
        sex,                        # 2. sex
        cp,                         # 3. cp
        HIDDEN_MEANS['trestbps'],   # 4. trestbps  ← auto-filled
        HIDDEN_MEANS['chol'],       # 5. chol       ← auto-filled
        HIDDEN_MEANS['fbs'],        # 6. fbs        ← auto-filled
        HIDDEN_MEANS['restecg'],    # 7. restecg    ← auto-filled
        thalach,                    # 8. thalach
        HIDDEN_MEANS['exang'],      # 9. exang      ← auto-filled
        oldpeak,                    # 10. oldpeak
        HIDDEN_MEANS['slope'],      # 11. slope     ← auto-filled
        ca,                         # 12. ca
        thal                        # 13. thal
    ]])
    # Shape is now (1, 13) ✅ — this is what fixes the error

    # Scale then predict
    input_scaled = scaler.transform(input_data)
    prediction   = model.predict(input_scaled)[0]
    probability  = model.predict_proba(input_scaled)[0][1]

    return int(prediction), round(float(probability) * 100, 1)


# ── 4. STREAMLIT UI ──────────────────────────────────────────

st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="centered"
)

# Header
st.markdown("## ❤️ Heart Disease Risk Predictor")
st.markdown("Fill in the fields below and click **Predict** to assess risk.")
st.divider()

# Input fields — only 7 shown to user
col1, col2 = st.columns(2)

with col1:
    age = st.number_input(
        "🎂 Age (years)",
        min_value=20, max_value=80, value=50, step=1
    )

    sex = st.selectbox(
        "⚧ Gender",
        options=[1, 0],
        format_func=lambda x: "Male" if x == 1 else "Female"
    )

    cp = st.selectbox(
        "💔 Chest Pain Type",
        options=[0, 1, 2, 3],
        format_func=lambda x: {
            0: "0 — Typical Angina",
            1: "1 — Atypical Angina",
            2: "2 — Non-Anginal Pain",
            3: "3 — Asymptomatic"
        }[x]
    )

    thalach = st.number_input(
        "💓 Max Heart Rate (bpm)",
        min_value=70, max_value=210, value=150, step=1
    )

with col2:
    ca = st.selectbox(
        "🩸 Major Vessels Coloured (0–3)",
        options=[0, 1, 2, 3]
    )

    oldpeak = st.number_input(
        "📉 ST Depression (oldpeak)",
        min_value=0.0, max_value=6.5, value=1.0, step=0.1,
        format="%.1f"
    )

    thal = st.selectbox(
        "🧬 Thalassemia",
        options=[1, 2, 3],
        format_func=lambda x: {
            1: "1 — Normal",
            2: "2 — Fixed Defect",
            3: "3 — Reversible Defect"
        }[x]
    )

st.divider()

# Predict button
if st.button("🔍 Predict", use_container_width=True, type="primary"):
    prediction, probability = predict(age, sex, cp, thalach, ca, oldpeak, thal)

    st.markdown("### 📊 Result")

    if prediction == 1:
        st.error(f"⚠️ **Heart Disease Detected**")
        st.metric("Risk Probability", f"{probability}%")
        st.warning(
            "This is a screening tool only. "
            "Please consult a cardiologist for a proper diagnosis."
        )
    else:
        st.success(f"✅ **No Heart Disease Detected**")
        st.metric("Risk Probability", f"{probability}%")
        st.info(
            "Low risk detected. Maintain a healthy lifestyle "
            "and schedule regular check-ups."
        )

    # Show what was auto-filled (transparency)
    with st.expander("ℹ️ See auto-filled values (used behind the scenes)"):
        st.write("These 6 fields were filled with dataset averages:")
        st.table({
            "Field"        : ["Resting BP", "Cholesterol", "Fasting Sugar", "Resting ECG", "Exercise Angina", "ST Slope"],
            "Value Used"   : [131.6,         246.5,         0.15,            0.53,           0.33,              1.40],
            "What it means": ["Average BP",  "Average chol","Avg fbs",       "Avg ECG",      "Avg exang",       "Avg slope"]
        })

# Footer
st.divider()
st.caption("ML Model: Heart Disease UCI Dataset | Course: SE-CD-638 | Dr. Aamir Arsalan")

