"""
Repo_7_Fraud_Detection — Home.py
Author : Mohamed · M3
"""
# streamlit run "E:\FINAL PROJECTS\P7_Fraud_Detection\Home.py"


import pathlib
import streamlit as st

st.set_page_config(page_title="Fraud Detection · M3", page_icon="🔍", layout="wide")

LOGO = pathlib.Path(__file__).parent / "M3_logo.png"

# ── SIDEBAR ──────────────────────────────────────────────────
with st.sidebar:
    if LOGO.exists():
        st.image(str(LOGO), width=70)
    st.markdown("### 🔍 Fraud Detection")
    st.markdown("M3 · ML Engine · P7")
    st.divider()
    st.markdown("**Navigate:**")
    st.markdown("📊 EDA Dashboard → 13 tabs")
    st.markdown("🤖 ML Models     → 5 tabs")

# ── CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stSidebar"]{background:#0f1923;}
[data-testid="stSidebar"] *{color:#e0e8f0 !important;}
.main{background:#f4f7fb;}
.hero{background:linear-gradient(135deg,#1a0a2e,#c62828);
      padding:48px 40px;border-radius:14px;margin-bottom:28px;}
.hero h1{color:#ffffff !important;font-size:2.4rem;font-weight:800;margin:0 0 8px 0;}
.hero p{color:#ffcdd2 !important;font-size:1.08rem;margin:0;}
.card{background:#ffffff;border-radius:10px;padding:22px 24px;
      box-shadow:0 2px 12px rgba(0,0,0,0.08);border-top:4px solid #c62828;}
.card h3{color:#c62828 !important;margin:0 0 8px 0;font-size:1.05rem;}
.card p{color:#37474f !important;font-size:0.92rem;margin:0;line-height:1.6;}
.stat-card{background:#ffffff;border-radius:10px;padding:18px;text-align:center;
           box-shadow:0 2px 10px rgba(0,0,0,0.07);border-bottom:3px solid #1565c0;}
.stat-num{font-size:1.9rem;font-weight:800;color:#c62828 !important;}
.stat-lbl{font-size:0.82rem;color:#546e7a !important;margin-top:4px;}
</style>""", unsafe_allow_html=True)

# ── HERO ─────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🔍 Fraud Detection Analysis</h1>
  <p>End-to-end ML pipeline · 51,000 transactions · Kaggle Dataset · M3 Portfolio · Project 7 of 12</p>
</div>""", unsafe_allow_html=True)

# ── STATS ROW ────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
stats = [
    ("51,000", "Transactions"),
    ("4.9%",   "Fraud Rate"),
    ("10",     "Features Used"),
    ("13",     "EDA Tabs"),
    ("12",     "ML Models"),
]
for col, (num, lbl) in zip([c1,c2,c3,c4,c5], stats):
    col.markdown(f"""
    <div class="stat-card">
      <div class="stat-num">{num}</div>
      <div class="stat-lbl">{lbl}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── ABOUT CARDS ──────────────────────────────────────────────
st.markdown("### 📌 About This Project")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="card">
    <h3>🎯 Objective</h3>
    <p>Detect fraudulent financial transactions using behavioral, temporal, and transactional signals.
    Tackle severe class imbalance (4.9% fraud) with class_weight='balanced' strategy.</p>
    </div>""", unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="card">
    <h3>📊 Dataset</h3>
    <p>Kaggle Fraud Detection · 51,000 transactions · 12 original features.
    Engineered: Hour, is_night, is_business_hours, high_velocity, new_account, Amount_Category.</p>
    </div>""", unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="card">
    <h3>🔑 Key Signals</h3>
    <p>Previous fraud history · Transaction velocity · Account age · Night-time transactions ·
    Unknown device · High transaction amounts.</p>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── EDA + ML OVERVIEW ────────────────────────────────────────
col4, col5 = st.columns(2)

with col4:
    st.markdown("### 📈 EDA Dashboard — 13 Tabs")
    tabs_eda = [
        ("1", "Data Overview",          "Shape, types, stats, data dictionary"),
        ("2", "Fraud Distribution ★",   "Fraud rate by type, payment method"),
        ("3", "Amount Analysis ★",      "Distribution + Pareto 80/20"),
        ("4", "Transaction Patterns ★", "Velocity + account age behavior"),
        ("5", "Time Analysis ★",        "Hourly fraud rate · night vs day"),
        ("6", "Device & Location ★",    "Fraud rate by device and city"),
        ("7", "Feature Engineering",    "Engineered flags + distributions"),
        ("8", "Correlation",            "Heatmap + top fraud predictors"),
        ("9", "Anomaly Detection ★",    "Z-score + IQR outlier analysis"),
        ("10","A/B Test ★",             "High velocity vs normal — Welch T-test"),
        ("11","Missing Values",         "Imputation strategy per column"),
        ("12","Multicollinearity",      "VIF analysis"),
        ("13","Insights & Report",      "Findings + recommendations + download"),
    ]
    for num, name, desc in tabs_eda:
        st.markdown(f"**Tab {num} · {name}** — {desc}")

with col5:
    st.markdown("### 🤖 ML Models — 5 Tabs")
    tabs_ml = [
        ("1", "Model Training",         "6 Regression + 6 Classification models"),
        ("2", "Regression Results",     "R², MAE, RMSE — predict Transaction_Amount"),
        ("3", "Classification Results", "F1, Precision, Recall, ROC-AUC — predict Fraudulent"),
        ("4", "Feature Importance",     "Top fraud predictors from best model"),
        ("5", "Predict",                "Interactive fraud risk scoring"),
    ]
    for num, name, desc in tabs_ml:
        st.markdown(f"**Tab {num} · {name}** — {desc}")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### ⚠️ Critical Note")
    st.error(
        "**Class Imbalance: 95.1% Legit / 4.9% Fraud**\n\n"
        "Standard accuracy is misleading. Evaluate with **F1, Precision, Recall, ROC-AUC**.\n\n"
        "All classifiers use `class_weight='balanced'`."
    )

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#90a4ae;font-size:0.85rem;'>"
    "Mohamed · M3 · ML Engine Portfolio · Project 7 of 12 · Fraud Detection</p>",
    unsafe_allow_html=True
)
