import streamlit as st
import joblib
import pandas as pd

model = joblib.load('breast_cancer_model.joblib')
scaler = joblib.load('breast_cancer_scaler.joblib')
label_encoder = joblib.load('breast_cancer_label_encoder.joblib')

st.title("Breast Cancer Diagnosis Prediction")

features = [
    'radius_mean', 'texture_mean', 'perimeter_mean',
    'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
    'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
    'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
    'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
    'fractal_dimension_se', 'radius_worst', 'texture_worst',
    'perimeter_worst', 'area_worst', 'smoothness_worst',
    'compactness_worst', 'concavity_worst', 'concave points_worst',
    'symmetry_worst', 'fractal_dimension_worst'
]

inputs = [st.number_input(f"Enter {feature}", value=0.0, format="%.5f") for feature in features]

if st.button("Predict"):
    input_df = pd.DataFrame([inputs], columns=features)
    scaled_data = scaler.transform(input_df)
    prediction = model.predict(scaled_data)[0]
    diagnosis = label_encoder.inverse_transform([prediction])[0]

    st.success(f"Predicted Diagnosis: {'Malignant (M)' if diagnosis == 'M' else 'Benign (B)'}")