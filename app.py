import streamlit as st
import pandas as pd
import joblib

# ---------- Page setup ----------
st.set_page_config(page_title="Abalone Age Prediction App", layout="wide")

# ---------- Load model (fast) ----------
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

# ---------- Simple styling ----------
st.markdown(
    """
    <style>
    .big-title {
        font-size: 48px;
        font-weight: 800;
        margin: 0.2rem 0 1.0rem 0;
    }
    .section-title {
        font-size: 28px;
        font-weight: 800;
        margin-top: 0.5rem;
        margin-bottom: 0.6rem;
    }
    .kpi-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 18px 18px;
    }
    .kpi-label {
        font-size: 16px;
        opacity: 0.8;
        margin-bottom: 6px;
    }
    .kpi-value {
        font-size: 44px;
        font-weight: 800;
        margin: 0;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------- Header ----------
st.markdown('<div class="big-title">🐚 Abalone Age Prediction App</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">🔢 Input Features</div>', unsafe_allow_html=True)

# ---------- Inputs in a form (faster, no rerun spam) ----------
with st.form("abalone_form"):
    c1, c2 = st.columns(2)

    # Gender (F/M/I like your screenshot)
    gender = st.radio("Gender", ["F", "M", "I"], horizontal=True)

    # Use ranges from your dataset (approx)
    with c1:
        length = st.number_input("Length", min_value=0.0, max_value=1.0, value=0.52, step=0.01, format="%.2f")
        height = st.number_input("Height", min_value=0.0, max_value=1.0, value=0.14, step=0.01, format="%.2f")
        shucked_weight = st.number_input("Shucked weight", min_value=0.0, max_value=2.0, value=0.22, step=0.01, format="%.2f")
        shell_weight = st.number_input("Shell weight", min_value=0.0, max_value=2.0, value=0.18, step=0.01, format="%.2f")

    with c2:
        diameter = st.number_input("Diameter", min_value=0.0, max_value=1.0, value=0.41, step=0.01, format="%.2f")
        whole_weight = st.number_input("Whole weight", min_value=0.0, max_value=4.0, value=0.83, step=0.01, format="%.2f")
        viscera_weight = st.number_input("Viscera weight", min_value=0.0, max_value=2.0, value=0.18, step=0.01, format="%.2f")

    predict = st.form_submit_button("🔮 Predict")

# ---------- Encode gender to match your training (get_dummies drop_first=True) ----------
# Your model uses columns: gender_I and gender_M (Female is baseline)
gender_I = 1 if gender == "I" else 0
gender_M = 1 if gender == "M" else 0

# ---------- Build input dataframe (must match training columns exactly) ----------
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

# ---------- Predict + Display ----------
if predict:
    rings = float(model.predict(input_df)[0])
    age = rings + 1.5

    st.success("✅ Prediction Complete")

    r1, r2 = st.columns(2)

    with r1:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Predicted Rings</div>
                <div class="kpi-value">{rings:.2f}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with r2:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">Predicted Age (years)</div>
                <div class="kpi-value">{age:.2f}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
