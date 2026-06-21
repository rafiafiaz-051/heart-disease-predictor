# ============================================================
#  Heart Disease Predictor — Streamlit App (Final Fixed)
#  Run:  streamlit run app.py
#  Files needed: champion_model.pkl + scaler.pkl
# ============================================================

import streamlit as st
import numpy as np
import joblib

# ── Load model & scaler ─────────────────────────────────────
model  = joblib.load('champion_model.pkl')
scaler = joblib.load('scaler.pkl')
N_FEATURES = scaler.n_features_in_

# ── NEUTRAL hidden defaults ──────────────────────────────────
# These are medians of the HEALTHY (no disease) class
# NOT overall means — avoids biasing toward disease
HIDDEN = {
    'trestbps': 130.0,   # healthy class median resting BP
    'chol'    : 234.0,   # healthy class median chol (< 240 = normal)
    'fbs'     : 0.0,     # 85% of people have fbs=0
    'restecg' : 0.0,     # 0 = normal ECG (most common healthy value)
    'exang'   : 0.0,     # 0 = no exercise angina (healthy default)
    'slope'   : 1.0,     # 1 = flat (most common value)
}

# ── PREDICTION FUNCTION ──────────────────────────────────────
def predict(age, sex, cp, thalach, ca, oldpeak, thal):

    # chol_high derived from hidden chol (234 < 240 → 0 = normal)
    chol_high = 1 if HIDDEN['chol'] > 240 else 0   # = 0 always now

    # age_group from actual user age
    if   age < 40: age_group = 0
    elif age < 55: age_group = 1
    elif age < 70: age_group = 2
    else:          age_group = 3

    # Build feature list in EXACT training order
    features = [
        age,                    # 1.  age
        sex,                    # 2.  sex
        cp,                     # 3.  cp
        HIDDEN['trestbps'],     # 4.  trestbps   ← hidden
        HIDDEN['chol'],         # 5.  chol        ← hidden
        HIDDEN['fbs'],          # 6.  fbs         ← hidden
        HIDDEN['restecg'],      # 7.  restecg     ← hidden
        thalach,                # 8.  thalach
        HIDDEN['exang'],        # 9.  exang       ← hidden
        oldpeak,                # 10. oldpeak
        HIDDEN['slope'],        # 11. slope       ← hidden
        ca,                     # 12. ca
        thal,                   # 13. thal
    ]

    # Add engineered features if scaler was fitted on 15
    if N_FEATURES == 15:
        features.append(age_group)   # 14. age_group_encoded
        features.append(chol_high)   # 15. chol_high = 0 (neutral)

    input_array  = np.array([features])          # shape (1, N) ✅
    input_scaled = scaler.transform(input_array)
    pred  = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0][1]

    return int(pred), round(float(proba) * 100, 1)


# ── UI ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="centered"
)

st.title("❤️ Heart Disease Risk Predictor")
st.caption("Enter patient details below. All fields are required.")
st.divider()

col1, col2 = st.columns(2)

with col1:
    age = st.number_input(
        "🎂 Age (years)",
        min_value=20, max_value=80, value=45, step=1,
        help="Patient age in years"
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
            3: "3 — No Chest Pain"
        }[x],
        help="Type of chest pain experienced"
    )

    thalach = st.number_input(
        "💓 Max Heart Rate (bpm)",
        min_value=70, max_value=210, value=160, step=1,
        help="Maximum heart rate achieved during exercise test"
    )

with col2:
    ca = st.selectbox(
        "🩸 Major Vessels Coloured (0–3)",
        options=[0, 1, 2, 3],
        help="Number of major vessels coloured by fluoroscopy (from angiography report)"
    )

    oldpeak = st.number_input(
        "📉 ST Depression (Oldpeak)",
        min_value=0.0, max_value=6.5, value=0.0,
        step=0.1, format="%.1f",
        help="ST depression induced by exercise relative to rest (from ECG/stress test)"
    )

    thal = st.selectbox(
        "🧬 Thalassemia Result",
        options=[1, 2, 3],
        format_func=lambda x: {
            1: "1 — Normal",
            2: "2 — Fixed Defect",
            3: "3 — Reversible Defect"
        }[x],
        help="Thalassemia blood disorder type from medical report"
    )

st.divider()

# ── Predict button ───────────────────────────────────────────
if st.button("🔍 Predict Risk", use_container_width=True, type="primary"):
    try:
        pred, proba = predict(age, sex, cp, thalach, ca, oldpeak, thal)

        st.subheader("📊 Prediction Result")

        if pred == 0:
            st.error("⚠️ **Heart Disease Risk Detected**")
            st.metric("Risk Probability", f"{proba}%",
                      delta="Above threshold", delta_color="inverse")
            st.warning(
                "⚕️ This is a **screening tool only**. "
                "Please consult a cardiologist for proper diagnosis."
            )
        else:
            st.success("✅ **Low Risk — No Heart Disease Detected**")
            st.metric("Risk Probability", f"{proba}%",
                      delta="Below threshold", delta_color="normal")
            st.info(
                "Regular health check-ups are still recommended. "
                "Maintain a healthy lifestyle."
            )

        # Progress bar for visual probability
        st.write("**Risk Level:**")
        st.progress(proba / 100)

        # Show what was auto-filled
        with st.expander("ℹ️ See auto-filled background values"):
            st.caption(
                "These 6 fields were not shown in the UI. "
                "They are filled with neutral/healthy baseline values "
                "to avoid biasing the prediction."
            )
            st.table({
                "Hidden Field"  : ["Resting BP", "Cholesterol", "Fasting Sugar",
                                   "Resting ECG", "Exercise Angina", "ST Slope"],
                "Value Used"    : [130.0, 234.0, 0.0, 0.0, 0.0, 1.0],
                "Why this value": [
                    "Healthy class median",
                    "Below 240 threshold (normal)",
                    "0 = no elevated sugar (85% of patients)",
                    "0 = normal ECG",
                    "0 = no angina (healthy default)",
                    "1 = flat slope (most common)"
                ]
            })

    except Exception as e:
        st.error(f"❌ Prediction error: {str(e)}")
        st.info(
            f"Scaler expects {N_FEATURES} features. "
            "Make sure champion_model.pkl and scaler.pkl are in the same folder."
        )

# ── Footer ───────────────────────────────────────────────────
st.divider()
st.caption(
    "SE-CD-638 Machine Learning | Dr. Aamir Arsalan | "
    "UCI Heart Disease Dataset | KNN Champion Model (AUC: 89.66%)"
)
