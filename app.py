import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Abalone Age Predictor", layout="centered")

@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

st.title("🐚 Abalone Age Prediction App")

with st.form("input_form"):
    length = st.slider("Length", 0.05, 0.9, 0.455)
    diameter = st.slider("Diameter", 0.05, 0.8, 0.365)
    height = st.slider("Height", 0.01, 0.3, 0.095)

    whole_weight = st.slider("Whole weight", 0.01, 3.0, 0.514)
    shucked_weight = st.slider("Shucked weight", 0.01, 1.5, 0.2245)
    viscera_weight = st.slider("Viscera weight", 0.01, 1.0, 0.101)
    shell_weight = st.slider("Shell weight", 0.01, 2.0, 0.15)

    gender = st.radio("Gender", ["Male", "Female", "Infant"])

    submit = st.form_submit_button("Predict")

gender_I = 1 if gender == "Infant" else 0
gender_M = 1 if gender == "Male" else 0

input_df = pd.DataFrame({
    "Length": [length],
    "Diameter": [diameter],
    "Height": [height],
    "Whole weight": [whole_weight],
    "Shucked weight": [shucked_weight],
    "Viscera weight": [viscera_weight],
    "Shell weight": [shell_weight],
    "gender_I": [gender_I],
    "gender_M": [gender_M]
})

st.dataframe(input_df)

if submit:
    rings = model.predict(input_df)[0]
    age = rings + 1.5
    st.success(f"Predicted Rings: {rings:.2f}")
    st.success(f"Estimated Age: {age:.2f} years")
