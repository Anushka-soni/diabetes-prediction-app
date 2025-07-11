# IMPORTANT STATEMENTS
import streamlit as st
import numpy as np
import joblib


# LOADING BOTH MODELS
rf_model = joblib.load('random_forest_model.sav')
gb_model = joblib.load('gradient_boosting_model.sav')


# You can hardcode or load actual trained accuracy scores
rf_accuracy = 90.13  # Update with your actual result
gb_accuracy = 92.11  # Update with your actual result


st.title("Diabetes Prediction System")

st.markdown("### Enter Patient Details:")


# INPUT FIELDS
pregnancies = st.number_input("Pregnancies", min_value=0)
glucose = st.number_input("Glucose", min_value=0)
bp = st.number_input("Blood Pressure", min_value=0)
skin = st.number_input("Skin Thickness", min_value=0)
insulin = st.number_input("Insulin", min_value=0)
bmi = st.number_input("BMI", min_value=0.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0)
age = st.number_input("Age", min_value=0)

if st.button("Predict with Both Models"):
    # PREPARING INPUT DATA
    input_data = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])

    # PREDICTIONS
    rf_prediction = rf_model.predict(input_data)[0]
    gb_prediction = gb_model.predict(input_data)[0]

    # RESULTS LAYOUTS SIDE-BY-SIDE
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Random Forest Classifier")
        st.write(f"**Accuracy:** {rf_accuracy}%")
        if rf_prediction == 0:
            st.success("Prediction: Not Diabetic")
        else:
            st.error("Prediction: Diabetic")

    with col2:
        st.subheader("Gradient Boosting Classifier")
        st.write(f"**Accuracy:** {gb_accuracy}%")
        if gb_prediction == 0:
            st.success("Prediction: Not Diabetic")
        else:
            st.error("Prediction: Diabetic")

