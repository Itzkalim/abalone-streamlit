import streamlit as st
import pandas as pd
import joblib

# =====================================================
# Page configuration
# =====================================================
st.set_page_config(
    page_title="Abalone Age Prediction App",
    layout="centered"
)

# =====================================================
# Load model (cached for speed)
# =====================================================
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

# =====================================================
# Custom CSS (Centered sticky header card + UI)
# =====================================================
st.markdown("""
<style>

/* ===== Center width ===== */
.block-container {
    max-width: 900px;
    padding-top: 1.2rem;
    padding-bottom: 2rem;
}

/* ===== Sticky Header Card (NOT full width) ===== */
.sticky-card {
    position: sticky;
    top: 12px;               /* a little gap from top */
    z-index: 999;
    background: linear-gradient(180deg, #0f172a 0%, #020617 100%);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 18px;
    padding: 16px 18px;
    box-shadow: 0 18px 35px rgba(0,0,0,0.45);
}

/* Sticky title text */
.sticky-title {
    font-size: 36px;
    font-weight: 900;
    margin: 0;
}

/* Space below sticky header so content doesn't overlap */
.after-sticky-gap {
    height: 18px;
}

/* ===== Main card styling ===== */
.center-card {
    background: linear-gradient(180deg, #0f172a 0%, #020617 100%);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 22px;
    padding: 34px 32px;
    box-shadow: 0 25px 45px rgba(0,0,0,0.45);
}

.section-title {
    font-size: 26px;
    font-weight: 800;
    margin-bottom: 22px;
}

/* Center Predict button */
.button-center {
    display: flex;
    justify-content: center;
    margin-top: 22px;
}

.stButton > button {
    width: 190px;
    height: 50px;
    border-radius: 14px;
    font-size: 16px;
    font-weight: 800;
}

/* Result cards */
.kpi-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 18px;
    padding: 22px;
    text-align: center;
}

.kpi-label {
    font-size: 15px;
    opacity: 0.8;
    margin-bottom: 6px;
}

.kpi-value {
    font-size: 42px;
    font-weight: 900;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# Sticky Header (Centered card style)
# =====================================================
st.markdown("""
<div class="sticky-card">
    <h1 class="sticky-title">🐚 Abalone Age Prediction App</h1>
</div>
<div class="after-sticky-gap"></div>
""", unsafe_allow_html=True)

# =====================================================
# Input Card
# =====================================================
st.markdown('<div class="center-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">🔢 Input Features</div>', unsafe_allow_html=True)

with st.form("abalone_form"):

    col1, col2 = st.columns(2)

    with col1:
        length = st.number_input("Length", min_value=0.0, max_value=1.0, value=0.52, step=0.01, format="%.2f")
        height = st.number_input("Height", min_value=0.0, max_value=1.0, value=0.14, step=0.01, format="%.2f")
        shucked_weight = st.number_input("Shucked weight", min_value=0.0, max_value=2.0, value=0.22, step=0.01, format="%.2f")
        shell_weight = st.number_input("Shell weight", min_value=0.0, max_value=2.0, value=0.18, step=0.01, format="%.2f")

    with col2:
        diameter = st.number_input("Diameter", min_value=0.0, max_value=1.0, value=0.41, step=0.01, format="%.2f")
        whole_weight = st.number_input("Whole weight", min_value=0.0, max_value=4.0, value=0.83, step=0.01, format="%.2f")
        viscera_weight = st.number_input("Viscera weight", min_value=0.0, max_value=2.0, value=0.18, step=0.01, format="%.2f")

    gender = st.radio("Gender", ["F", "M", "I"], horizontal=True)

    st.markdown('<div class="button-center">', unsafe_allow_html=True)
    submit = st.form_submit_button("🔮 Predict")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# =====================================================
# Encode gender (matches training: get_dummies drop_first=True)
# Female is baseline => only gender_I and gender_M exist
# =====================================================
gender_I = 1 if gender == "I" else 0
gender_M = 1 if gender == "M" else 0

# =====================================================
# Prediction + Results
# =====================================================
if submit:
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
