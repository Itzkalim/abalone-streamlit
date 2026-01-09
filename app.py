import streamlit as st
import pandas as pd
import joblib

# =====================================================
# Page configuration (wide helps avoid scrolling)
# =====================================================
st.set_page_config(
    page_title="Abalone Age Prediction App",
    layout="wide"
)

# =====================================================
# Load model (cached for speed)
# =====================================================
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

# =====================================================
# Custom CSS (sticky header + compact one-page UI)
# =====================================================
st.markdown("""
<style>

/* Make page compact to avoid scroll */
.block-container {
    max-width: 1200px;
    padding-top: 0.8rem;
    padding-bottom: 0.8rem;
}

/* Sticky Header Card (centered) */
.sticky-card {
    position: sticky;
    top: 10px;
    z-index: 999;
    background: linear-gradient(180deg, #0f172a 0%, #020617 100%);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 18px;
    padding: 14px 18px;
    box-shadow: 0 18px 35px rgba(0,0,0,0.45);
    margin-bottom: 14px;
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
    padding: 22px 22px;
    box-shadow: 0 20px 40px rgba(0,0,0,0.40);
}

/* Smaller section title */
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
    margin-top: 14px;
}

/* KPI Cards */
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
    margin: 0;
}

/* Reduce spacing between elements to fit one screen */
div[data-testid="stVerticalBlock"] > div:has(> div[data-testid="stForm"]) {
    gap: 0.6rem;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# Sticky Header (always visible)
# =====================================================
st.markdown("""
<div class="sticky-card">
    <h1 class="sticky-title">🐚 Abalone Age Prediction App</h1>
</div>
""", unsafe_allow_html=True)

# =====================================================
# Two-column layout: Inputs (Left) | Results (Right)
# =====================================================
left, right = st.columns([1.1, 0.9], gap="large")

# Keep prediction values
rings = None
age = None
predicted = False

# ================= LEFT: INPUTS =================
with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🔢 Input Features</div>', unsafe_allow_html=True)

    with st.form("abalone_form"):

        c1, c2 = st.columns(2)

        with c1:
            length = st.number_input("Length", min_value=0.0, max_value=1.0, value=0.52, step=0.01, format="%.2f")
            height = st.number_input("Height", min_value=0.0, max_value=1.0, value=0.14, step=0.01, format="%.2f")
            shucked_weight = st.number_input("Shucked weight", min_value=0.0, max_value=2.0, value=0.22, step=0.01, format="%.2f")
            shell_weight = st.number_input("Shell weight", min_value=0.0, max_value=2.0, value=0.18, step=0.01, format="%.2f")

        with c2:
            diameter = st.number_input("Diameter", min_value=0.0, max_value=1.0, value=0.41, step=0.01, format="%.2f")
            whole_weight = st.number_input("Whole weight", min_value=0.0, max_value=4.0, value=0.83, step=0.01, format="%.2f")
            viscera_weight = st.number_input("Viscera weight", min_value=0.0, max_value=2.0, value=0.18, step=0.01, format="%.2f")

        gender = st.radio("Gender", ["F", "M", "I"], horizontal=True)

        st.markdown('<div class="button-center">', unsafe_allow_html=True)
        submit = st.form_submit_button("🔮 Predict")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ================= RIGHT: RESULTS =================
with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📌 Prediction Output</div>', unsafe_allow_html=True)

    # Default message
    if "last_pred" not in st.session_state:
        st.info("Enter values and click **Predict** to see results here.")

    # When Predict pressed, compute and store results
    if 'submit' in locals() and submit:
        # Encode gender (drop_first=True => Female baseline)
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

        st.session_state["last_pred"] = {"rings": rings, "age": age}

        st.success("✅ Prediction Complete")

    # Show last prediction (so it stays visible without scrolling)
    if "last_pred" in st.session_state:
        rings = st.session_state["last_pred"]["rings"]
        age = st.session_state["last_pred"]["age"]

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

        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("Show input used (optional)"):
            st.write("These were the last inputs used for prediction:")
            st.write({
                "Length": length,
                "Diameter": diameter,
                "Height": height,
                "Whole weight": whole_weight,
                "Shucked weight": shucked_weight,
                "Viscera weight": viscera_weight,
                "Shell weight": shell_weight,
                "Gender": gender
            })

    st.markdown('</div>', unsafe_allow_html=True)
