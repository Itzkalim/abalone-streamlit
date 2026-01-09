import streamlit as st
import pandas as pd
import joblib

# =====================================================
# Page configuration
# =====================================================
st.set_page_config(
    page_title="Abalone Age Prediction App",
    layout="wide"
)

# =====================================================
# Load model (cached)
# =====================================================
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

# =====================================================
# CSS — REMOVE STREAMLIT TOP BARS + STYLE APP
# =====================================================
st.markdown("""
<style>

/* 🚫 REMOVE STREAMLIT DEFAULT UI */
header {visibility: hidden;}
footer {visibility: hidden;}
[data-testid="stToolbar"] {display: none;}
[data-testid="stDecoration"] {display: none;}
[data-testid="stStatusWidget"] {display: none;}

/* Layout compact */
.block-container {
    max-width: 1200px;
    padding-top: 0.5rem;
    padding-bottom: 0.5rem;
}

/* Sticky centered header card */
.sticky-card {
    position: sticky;
    top: 10px;
    z-index: 999;
    background: linear-gradient(180deg, #0f172a 0%, #020617 100%);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 18px;
    padding: 14px 20px;
    box-shadow: 0 18px 35px rgba(0,0,0,0.45);
    margin-bottom: 12px;
}

.sticky-title {
    font-size: 34px;
    font-weight: 900;
    margin: 0;
}

/* Cards */
.card {
    background: linear-gradient(180deg, #0f172a 0%, #020617 100%);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 20px;
    padding: 22px;
    box-shadow: 0 20px 40px rgba(0,0,0,0.40);
}

/* Section titles */
.section-title {
    font-size: 22px;
    font-weight: 800;
    margin-bottom: 14px;
}

/* Button */
.stButton > button {
    width: 190px;
    height: 46px;
    border-radius: 14px;
    font-size: 16px;
    font-weight: 800;
}

.button-center {
    display: flex;
    justify-content: center;
    margin-top: 12px;
}

/* KPI cards */
.kpi-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 18px;
    padding: 18px;
    text-align: center;
}

.kpi-label {
    font-size: 14px;
    opacity: 0.85;
    margin-bottom: 6px;
}

.kpi-value {
    font-size: 38px;
    font-weight: 900;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# Sticky Header (only your header, nothing else)
# =====================================================
st.markdown("""
<div class="sticky-card">
    <h1 class="sticky-title">🐚 Abalone Age Prediction App</h1>
</div>
""", unsafe_allow_html=True)

# =====================================================
# Layout: Inputs | Results (ONE PAGE)
# =====================================================
left, right = st.columns([1.1, 0.9], gap="large")

# ================= INPUTS =================
with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🔢 Input Features</div>', unsafe_allow_html=True)

    with st.form("abalone_form"):

        c1, c2 = st.columns(2)

        with c1:
            length = st.number_input("Length", 0.0, 1.0, 0.52, 0.01)
            height = st.number_input("Height", 0.0, 1.0, 0.14, 0.01)
            shucked_weight = st.number_input("Shucked weight", 0.0, 2.0, 0.22, 0.01)
            shell_weight = st.number_input("Shell weight", 0.0, 2.0, 0.18, 0.01)

        with c2:
            diameter = st.number_input("Diameter", 0.0, 1.0, 0.41, 0.01)
            whole_weight = st.number_input("Whole weight", 0.0, 4.0, 0.83, 0.01)
            viscera_weight = st.number_input("Viscera weight", 0.0, 2.0, 0.18, 0.01)

        gender = st.radio("Gender", ["F", "M", "I"], horizontal=True)

        st.markdown('<div class="button-center">', unsafe_allow_html=True)
        submit = st.form_submit_button("🔮 Predict")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ================= RESULTS =================
with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📌 Prediction Output</div>', unsafe_allow_html=True)

    if submit:
        gender_I = 1 if gender == "I" else 0
        gender_M = 1 if gender == "M" else 0

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

        k1, k2 = st.columns(2)

        with k1:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">Predicted Rings</div>
                    <div class="kpi-value">{rings:.2f}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with k2:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">Predicted Age (years)</div>
                    <div class="kpi-value">{age:.2f}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    else:
        st.info("Enter inputs and click **Predict** to see results.")

    st.markdown('</div>', unsafe_allow_html=True)
