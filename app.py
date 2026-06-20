# ============================================================
#  Heart Disease Predictor — Streamlit App (Fixed v2)
#  Run:  streamlit run app.py
# ============================================================

import streamlit as st
import numpy as np
import joblib

# ── Load model & scaler ──────────────────────────────────────
model  = joblib.load('champion_model.pkl')
scaler = joblib.load('scaler.pkl')

# ── Check exactly how many features scaler expects ───────────
N_FEATURES = scaler.n_features_in_
st.sidebar.write(f"DEBUG — scaler expects: {N_FEATURES} features")

# ── Mean values for ALL hidden fields ────────────────────────
# These cover both original 13 + 2 engineered features
HIDDEN_MEANS = {
    'trestbps'          : 131.6,
    'chol'              : 246.5,
    'fbs'               : 0.15,
    'restecg'           : 0.53,
    'exang'             : 0.33,
    'slope'             : 1.40,
    'age_group_encoded' : 1.8,   # engineered in Phase 1
    'chol_high'         : 0.37,  # engineered in Phase 1
}

# ── PREDICTION FUNCTION ──────────────────────────────────────
def predict(age, sex, cp, thalach, ca, oldpeak, thal):

    # Base 13 original features in training order
    features_13 = [
        age,                        # 1.  age
        sex,                        # 2.  sex
        cp,                         # 3.  cp
        HIDDEN_MEANS['trestbps'],   # 4.  trestbps   ← auto-filled
        HIDDEN_MEANS['chol'],       # 5.  chol        ← auto-filled
        HIDDEN_MEANS['fbs'],        # 6.  fbs         ← auto-filled
        HIDDEN_MEANS['restecg'],    # 7.  restecg     ← auto-filled
        thalach,                    # 8.  thalach
        HIDDEN_MEANS['exang'],      # 9.  exang       ← auto-filled
        oldpeak,                    # 10. oldpeak
        HIDDEN_MEANS['slope'],      # 11. slope       ← auto-filled
        ca,                         # 12. ca
        thal                        # 13. thal
    ]

    # Add engineered features if scaler expects 15
    if N_FEATURES == 15:
        # age_group_encoded: 0=Young(<40), 1=Middle(40-55), 2=Senior(55-70), 3=Elderly(70+)
        if   age < 40:  age_group = 0
        elif age < 55:  age_group = 1
        elif age < 70:  age_group = 2
        else:           age_group = 3

        chol_high = 1 if HIDDEN_MEANS['chol'] > 240 else 0

        features_13.append(age_group)   # 14. age_group_encoded
        features_13.append(chol_high)   # 15. chol_high

    input_data   = np.array([features_13])          # shape (1, N_FEATURES) ✅
    input_scaled = scaler.transform(input_data)
    prediction   = model.predict(input_scaled)[0]
    probability  = model.predict_proba(input_scaled)[0][1]

    return int(prediction), round(float(probability) * 100, 1)


# ── STREAMLIT UI ─────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="centered"
)

st.markdown("## ❤️ Heart Disease Risk Predictor")
st.markdown("Fill in the fields below and click **Predict** to assess risk.")
st.divider()

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
        min_value=0.0, max_value=6.5, value=1.0,
        step=0.1, format="%.1f"
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

if st.button("🔍 Predict", use_container_width=True, type="primary"):
    try:
        prediction, probability = predict(age, sex, cp, thalach, ca, oldpeak, thal)

        st.markdown("### 📊 Result")
        if prediction == 1:
            st.error("⚠️ **Heart Disease Detected**")
            st.metric("Risk Probability", f"{probability}%")
            st.warning("This is a screening tool only. Please consult a cardiologist.")
        else:
            st.success("✅ **No Heart Disease Detected**")
            st.metric("Risk Probability", f"{probability}%")
            st.info("Low risk detected. Maintain a healthy lifestyle and schedule regular check-ups.")

        with st.expander("ℹ️ See auto-filled values (used behind the scenes)"):
            st.write(f"Model uses **{N_FEATURES} features** total. These were auto-filled:")
            rows = {
                "Field"     : ["Resting BP", "Cholesterol", "Fasting Sugar",
                               "Resting ECG", "Exercise Angina", "ST Slope"],
                "Value Used": [131.6, 246.5, 0.15, 0.53, 0.33, 1.40]
            }
            if N_FEATURES == 15:
                rows["Field"].extend(["Age Group (encoded)", "High Cholesterol Flag"])
                rows["Value Used"].extend(["Computed from age", "0 (avg chol < 240)"])
            st.table(rows)

    except Exception as e:
        st.error(f"Prediction error: {e}")
        st.info(
            f"The scaler expects {N_FEATURES} features. "
            "Check the sidebar — if this number looks wrong, "
            "re-run Phase 1 notebook and re-save scaler.pkl."
        )

st.divider()
st.caption("ML Model: Heart Disease UCI Dataset | Course: SE-CD-638 | Dr. Aamir Arsalan")


