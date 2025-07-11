# IMPORTANT STATEMENTS
import streamlit as st
import numpy as np
import joblib

# Load both models
rf_model = joblib.load('random_forest_model.sav')
gb_model = joblib.load('gradient_boosting_model.sav')

# Accuracy scores (hardcoded for display)
rf_accuracy = 90.13
gb_accuracy = 92.11

# --- PAGE CONFIG ---
st.set_page_config(page_title="Diabetes Predictor", layout="centered")

# --- HEADER SECTION ---
st.markdown("""
    <div style='text-align: center;'>
        <h1 style='color:#4CAF50;'>🩺 Diabetes Prediction System</h1>
        <p style='font-size:18px;'>Enter the patient's medical info below and get instant predictions from two ML models</p>
    </div>
""", unsafe_allow_html=True)

st.markdown("---")

# --- INPUT FORM ---
with st.form("input_form"):
    col1, col2 = st.columns(2)

    with col1:
        pregnancies = st.number_input("Pregnancies", min_value=0)
        glucose = st.number_input("Glucose", min_value=0)
        bp = st.number_input("Blood Pressure", min_value=0)
        skin = st.number_input("Skin Thickness", min_value=0)

    with col2:
        insulin = st.number_input("Insulin", min_value=0)
        bmi = st.number_input("BMI", min_value=0.0, format="%.2f")
        dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, format="%.2f")
        age = st.number_input("Age", min_value=0)

    submitted = st.form_submit_button("🔍 Predict with Both Models")

# --- PREDICTION ---
if submitted:
    input_data = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
    rf_prediction = rf_model.predict(input_data)[0]
    gb_prediction = gb_model.predict(input_data)[0]

    # Layout for results
    st.markdown("### 📊 Prediction Results")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🌲 Random Forest")
        st.write(f"**Accuracy:** {rf_accuracy}%")
        if rf_prediction == 0:
            st.success("✅ Not Diabetic")
        else:
            st.error("❌ Diabetic")

    with col2:
        st.markdown("#### 🚀 Gradient Boosting")
        st.write(f"**Accuracy:** {gb_accuracy}%")
        if gb_prediction == 0:
            st.success("✅ Not Diabetic")
        else:
            st.error("❌ Diabetic")

    st.markdown("---")
    st.info("Note: Different models may provide different results based on their internal logic and training.")

