"""
Repo_7_Fraud_Detection — EDA_dashboard.py  (13 Tabs)
Author : Mohamed · M3
Dataset: Kaggle Fraud Detection · 51,000 transactions
"""
# streamlit run "E:\FINAL PROJECTS\P7_Fraud_Detection\P7_EDA_dashboard.py"

import pathlib, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from scipy.stats import zscore
from statsmodels.stats.outliers_influence import variance_inflation_factor
import streamlit as st

warnings.filterwarnings("ignore")
S = st.session_state

# ── PAGE CONFIG ──────────────────────────────────────────────
st.set_page_config(page_title="EDA · Fraud Detection · M3",
                   page_icon="🔍", layout="wide")

# ── PATHS ────────────────────────────────────────────────────
LOGO = pathlib.Path(__file__).parent.parent / "M3_logo.png"
DATA = pathlib.Path(__file__).parent.parent / "data" / "fraud_clean.csv"

# ── SIDEBAR ──────────────────────────────────────────────────
with st.sidebar:
    if LOGO.exists():
        st.image(str(LOGO), width=70)
    st.markdown("### 🔍 EDA Dashboard")
    st.markdown("Fraud Detection · 13 Tabs")
    st.divider()
    st.markdown("### 📂 Dataset")
    _uploaded = st.file_uploader("Upload Clean CSV", type=["csv"],
                                  help="Upload fraud_clean.csv from Jupyter.\n"
                                       "Leave empty to use default dataset.",
                                  key="eda_upload")
    if _uploaded is not None:
        st.success(f"✅ Using: {_uploaded.name}")
    else:
        st.info("Using default: fraud_clean.csv")

# ── PALETTE ──────────────────────────────────────────────────
CLR = {"primary":"#1565c0","success":"#2e7d32","warning":"#e65100",
       "danger":"#c62828","teal":"#00695c","accent":"#00695c",
       "secondary":"#455a64","light":"#e3f2fd","dark":"#1a237e",
       "purple":"#6a1b9a","amber":"#f57f17","pink":"#ad1457",
       "indigo":"#283593","cyan":"#00838f","lime":"#558b2f",
       "brown":"#4e342e","grey":"#546e7a","white":"#ffffff","black":"#212121"}

# ── CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stSidebar"]{background:#0f1923;}
[data-testid="stSidebar"] *{color:#e0e8f0 !important;}
[data-testid="stSidebar"] [data-testid="stFileUploader"]{background:#1a2633;border:1.5px dashed #4a7fa5;border-radius:8px;padding:6px;}
[data-testid="stSidebar"] [data-testid="stFileUploader"] *{color:#e0e8f0 !important;}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"]{background:#1a2633 !important;border:none !important;}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzoneInstructions"] *{color:#a0bcd4 !important;}
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"]{background:#1565c0 !important;color:#ffffff !important;border:none !important;border-radius:6px !important;}
.main{background:#f4f7fb;}
div[data-testid="metric-container"]{background:#e3f2fd;border-left:4px solid #1565c0;
  border-radius:6px;padding:10px 14px;}
.sec-header{background:linear-gradient(90deg,#1565c0,#c62828);color:#ffffff !important;
  padding:10px 18px;border-radius:8px;font-size:1.1rem;font-weight:700;margin-bottom:16px;}
.insight-box{background:#e8f5e9;border-left:4px solid #2e7d32;padding:12px 16px;
  border-radius:0 6px 6px 0;margin:8px 0;}
.insight-box p{color:#1b3a1f !important;margin:0;font-size:0.93rem;line-height:1.6;}
.warn-box{background:#fff3e0;border-left:4px solid #e65100;padding:12px 16px;
  border-radius:0 6px 6px 0;margin:8px 0;}
.warn-box p{color:#4a2000 !important;margin:0;font-size:0.93rem;line-height:1.6;}
.info-box{background:#e3f2fd;border-left:4px solid #1565c0;padding:12px 16px;
  border-radius:0 6px 6px 0;margin:8px 0;}
.info-box p{color:#0d2a4a !important;margin:0;font-size:0.93rem;line-height:1.6;}
.fraud-box{background:#fce4ec;border-left:4px solid #c62828;padding:12px 16px;
  border-radius:0 6px 6px 0;margin:8px 0;}
.fraud-box p{color:#4a0000 !important;margin:0;font-size:0.93rem;line-height:1.6;}
</style>""", unsafe_allow_html=True)

# ── HELPERS ──────────────────────────────────────────────────
def sec(t): st.markdown(f'<div class="sec-header">{t}</div>', unsafe_allow_html=True)
def insight(t): st.markdown(f'<div class="insight-box"><p>✅ {t}</p></div>', unsafe_allow_html=True)
def warn(t):    st.markdown(f'<div class="warn-box"><p>⚠️ {t}</p></div>', unsafe_allow_html=True)
def info(t):    st.markdown(f'<div class="info-box"><p>ℹ️ {t}</p></div>', unsafe_allow_html=True)
def fraud_alert(t): st.markdown(f'<div class="fraud-box"><p>🚨 {t}</p></div>', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# LOAD DATA
# ════════════════════════════════════════════════════════════
@st.cache_data
def load_data(file_bytes=None) -> pd.DataFrame:
    import io as _io
    if file_bytes is not None:
        df = pd.read_csv(_io.BytesIO(file_bytes), sep=",", decimal=".")
    else:
        df = pd.read_csv(DATA, sep=",", decimal=".")
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    df.columns = df.columns.str.strip()
    return df

_up = S.get("eda_upload", None)
if _up is not None:
    _bytes = _up.read(); _up.seek(0)
    df = load_data(file_bytes=_bytes)
    S["file_name"] = _up.name
else:
    df = load_data()
    S["file_name"] = "fraud_clean.csv"

if df.empty:
    st.warning("⚠️ No data found. Upload fraud_clean.csv or run P7_clean_data.py first.")
    st.stop()

S["df_work"] = df

# ── Column groups ─────────────────────────────────────────────
NUM_COLS = [c for c in ["Transaction_Amount","Previous_Fraudulent_Transactions",
                         "Account_Age","Number_of_Transactions_Last_24H","Hour"]
            if c in df.columns]
CAT_COLS = [c for c in ["Transaction_Type","Device_Used","Location","Payment_Method"]
            if c in df.columns]
ENG_COLS = [c for c in ["is_night","is_business_hours","high_velocity","new_account"]
            if c in df.columns]
ENC_COLS = [c for c in df.columns if c.endswith("_enc")]
TARGET   = "Fraudulent"

df_fraud = df[df[TARGET] == 1]
df_legit = df[df[TARGET] == 0]
fraud_rate = len(df_fraud) / len(df) * 100

# ════════════════════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════════════════════
tabs = st.tabs([
    "1 · Data Overview",
    "2 · Fraud Distribution ★",
    "3 · Amount Analysis ★",
    "4 · Transaction Patterns ★",
    "5 · Time Analysis ★",
    "6 · Device & Location ★",
    "7 · Feature Engineering",
    "8 · Correlation",
    "9 · Anomaly Detection ★",
    "10 · A/B Test ★",
    "11 · Missing Values",
    "12 · Multicollinearity",
    "13 · Insights & Report",
])

# ════════════════════════════════════════════════════════════
# TAB 1 — DATA OVERVIEW
# ════════════════════════════════════════════════════════════
with tabs[0]:
    sec("📋 Tab 1 — Data Overview")

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Total Transactions", f"{len(df):,}")
    c2.metric("Fraudulent",         f"{len(df_fraud):,}")
    c3.metric("Legitimate",         f"{len(df_legit):,}")
    c4.metric("Fraud Rate",         f"{fraud_rate:.1f}%")
    c5.metric("Features",           f"{df.shape[1]}")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        sec("📄 First 10 Rows")
        st.dataframe(df.head(10), use_container_width=True)
    with col2:
        sec("📐 Column Info")
        info_df = pd.DataFrame({
            "Column": df.columns,
            "Dtype":  df.dtypes.astype(str).values,
            "Nulls":  df.isnull().sum().values,
            "Null%":  (df.isnull().mean()*100).round(1).values,
        })
        st.dataframe(info_df, use_container_width=True)

    st.markdown("---")
    sec("📊 Descriptive Statistics")
    st.dataframe(df[NUM_COLS].describe().round(3), use_container_width=True)

    st.markdown("---")
    sec("🗂 Data Dictionary")
    dd = pd.DataFrame({
        "Column": ["Transaction_Amount","Transaction_Type","Hour","Device_Used",
                   "Location","Previous_Fraudulent_Transactions","Account_Age",
                   "Number_of_Transactions_Last_24H","Payment_Method","Fraudulent",
                   "is_night","is_business_hours","high_velocity","new_account","Amount_Category"],
        "Type": ["Numeric","Categorical","Numeric","Categorical","Categorical",
                 "Numeric","Numeric","Numeric","Categorical","Target (binary)",
                 "Engineered","Engineered","Engineered","Engineered","Engineered"],
        "Description": [
            "Transaction value in USD",
            "ATM Withdrawal / POS Payment / Online Purchase / Bill Payment / Bank Transfer",
            "Hour of transaction (0–23), renamed from Time_of_Transaction",
            "Tablet / Mobile / Desktop / Unknown / Unknown Device",
            "City where transaction occurred",
            "Count of user's past fraudulent transactions — strongest predictor ⭐",
            "Account age in days",
            "Transactions made by this user in the last 24 hours — velocity signal ⭐",
            "Debit Card / Credit Card / UPI / Net Banking / Unknown",
            "1 = Fraudulent · 0 = Legitimate",
            "1 if transaction between 10 PM – 6 AM",
            "1 if transaction between 9 AM – 5 PM",
            "1 if Number_of_Transactions_Last_24H > 75th percentile",
            "1 if Account_Age < 25th percentile",
            "Amount range: Very Low / Low / Medium / High / Very High",
        ]
    })
    st.dataframe(dd, use_container_width=True)

    fraud_alert(f"Only {fraud_rate:.1f}% fraud — severe imbalance. "
                "Accuracy is misleading. Use F1, Precision, Recall, ROC-AUC.")
    info("Transaction_ID and User_ID dropped — no predictive value.")

# ════════════════════════════════════════════════════════════
# TAB 2 — FRAUD DISTRIBUTION ★
# ════════════════════════════════════════════════════════════
with tabs[1]:
    sec("🎯 Tab 2 — Fraud Distribution ★")
    info("Most critical tab — where is fraud concentrated by category?")

    col1, col2 = st.columns(2)
    with col1:
        sec("📊 Class Balance")
        bal = pd.DataFrame({"Label":["Legitimate","Fraudulent"],
                            "Count":[len(df_legit), len(df_fraud)]})
        bal["Pct"] = (bal["Count"]/len(df)*100).round(2)
        fig = px.bar(bal, x="Label", y="Count", color="Label",
                     color_discrete_map={"Legitimate":CLR["primary"],"Fraudulent":CLR["danger"]},
                     text=bal["Pct"].apply(lambda x: f"{x}%"),
                     title="Fraudulent vs Legitimate")
        fig.update_traces(textposition="outside")
        fig.update_layout(height=370, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        sec("🥧 Proportion")
        fig2 = px.pie(bal, names="Label", values="Count",
                      color="Label",
                      color_discrete_map={"Legitimate":CLR["primary"],"Fraudulent":CLR["danger"]},
                      title="Class Distribution", hole=0.45)
        fig2.update_traces(textinfo="percent+label")
        fig2.update_layout(height=370)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    col3, col4 = st.columns(2)
    with col3:
        sec("📊 Fraud Rate by Transaction Type")
        ft = df.groupby("Transaction_Type")[TARGET].agg(Total="count",Fraud="sum").reset_index()
        ft["Fraud_Rate%"] = (ft["Fraud"]/ft["Total"]*100).round(2)
        ft = ft.sort_values("Fraud_Rate%", ascending=False)
        fig3 = px.bar(ft, x="Transaction_Type", y="Fraud_Rate%",
                      color="Fraud_Rate%", color_continuous_scale=["#2e7d32","#e65100","#c62828"],
                      title="Fraud Rate % by Transaction Type",
                      text=ft["Fraud_Rate%"].apply(lambda x: f"{x:.1f}%"))
        fig3.add_hline(y=fraud_rate, line_dash="dash", line_color="blue",
                       annotation_text=f"Avg {fraud_rate:.1f}%")
        fig3.update_traces(textposition="outside")
        fig3.update_layout(height=370)
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        sec("📊 Fraud Rate by Payment Method")
        pm = df.groupby("Payment_Method")[TARGET].agg(Total="count",Fraud="sum").reset_index()
        pm["Fraud_Rate%"] = (pm["Fraud"]/pm["Total"]*100).round(2)
        pm = pm.sort_values("Fraud_Rate%", ascending=False)
        fig4 = px.bar(pm, x="Payment_Method", y="Fraud_Rate%",
                      color="Fraud_Rate%", color_continuous_scale=["#2e7d32","#e65100","#c62828"],
                      title="Fraud Rate % by Payment Method",
                      text=pm["Fraud_Rate%"].apply(lambda x: f"{x:.1f}%"))
        fig4.add_hline(y=fraud_rate, line_dash="dash", line_color="blue",
                       annotation_text=f"Avg {fraud_rate:.1f}%")
        fig4.update_traces(textposition="outside")
        fig4.update_layout(height=370)
        st.plotly_chart(fig4, use_container_width=True)

    fraud_alert(f"Fraud rate {fraud_rate:.1f}% — class_weight='balanced' mandatory in all ML models.")
    insight("Channels with above-average fraud rate are highest-priority for additional verification rules.")
    warn("'Unknown' payment method shows elevated fraud — missing data itself is a fraud signal.")

# ════════════════════════════════════════════════════════════
# TAB 3 — AMOUNT ANALYSIS ★
# ════════════════════════════════════════════════════════════
with tabs[2]:
    sec("💰 Tab 3 — Transaction Amount Analysis ★")
    info("Amount is a strong fraud signal — distribution shape and Pareto analysis reveal targeting strategy.")

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Fraud — Avg Amount",  f"${df_fraud['Transaction_Amount'].mean():,.0f}")
    c2.metric("Legit — Avg Amount",  f"${df_legit['Transaction_Amount'].mean():,.0f}")
    c3.metric("Fraud — Median",      f"${df_fraud['Transaction_Amount'].median():,.0f}")
    c4.metric("Legit — Median",      f"${df_legit['Transaction_Amount'].median():,.0f}")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        sec("📊 Amount Distribution — Fraud vs Legit")
        fig, ax = plt.subplots(figsize=(8,4))
        ax.hist(df_legit["Transaction_Amount"], bins=50, alpha=0.6,
                color=CLR["primary"], label="Legitimate", density=True)
        ax.hist(df_fraud["Transaction_Amount"], bins=50, alpha=0.7,
                color=CLR["danger"], label="Fraudulent", density=True)
        ax.axvline(df_legit["Transaction_Amount"].mean(), color=CLR["primary"],
                   lw=2, ls="--", label=f"Legit Mean=${df_legit['Transaction_Amount'].mean():.0f}")
        ax.axvline(df_fraud["Transaction_Amount"].mean(), color=CLR["danger"],
                   lw=2, ls="--", label=f"Fraud Mean=${df_fraud['Transaction_Amount'].mean():.0f}")
        ax.set_xlabel("Amount ($)"); ax.set_ylabel("Density")
        ax.set_title("Amount: Fraud vs Legitimate"); ax.legend(fontsize=8)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col2:
        sec("📦 Box Plot by Class")
        fig2, ax2 = plt.subplots(figsize=(6,4))
        bp = ax2.boxplot([df_legit["Transaction_Amount"].dropna(),
                          df_fraud["Transaction_Amount"].dropna()],
                         patch_artist=True, labels=["Legitimate","Fraudulent"])
        bp["boxes"][0].set_facecolor(CLR["light"])
        bp["boxes"][1].set_facecolor("#fce4ec")
        for m in bp["medians"]: m.set_color(CLR["danger"]); m.set_linewidth(2)
        ax2.set_ylabel("Amount ($)"); ax2.set_title("Amount by Class")
        plt.tight_layout(); st.pyplot(fig2); plt.close()

    st.markdown("---")
    sec("📊 Fraud Rate by Amount Category")
    if "Amount_Category" in df.columns:
        fa = df.groupby("Amount_Category", observed=True)[TARGET].agg(
            Total="count", Fraud="sum").reset_index()
        fa["Fraud_Rate%"] = (fa["Fraud"]/fa["Total"]*100).round(2)
        col3, col4 = st.columns(2)
        with col3:
            st.dataframe(fa, use_container_width=True)
        with col4:
            fig3 = px.bar(fa, x="Amount_Category", y="Fraud_Rate%",
                          color="Fraud_Rate%",
                          color_continuous_scale=["#2e7d32","#e65100","#c62828"],
                          title="Fraud Rate % by Amount Range",
                          text=fa["Fraud_Rate%"].apply(lambda x: f"{x:.1f}%"))
            fig3.update_traces(textposition="outside")
            fig3.update_layout(height=350)
            st.plotly_chart(fig3, use_container_width=True)

    st.markdown("---")
    sec("📈 Pareto — 80/20 Rule on Fraud Amounts ★")
    fraud_amounts = df_fraud["Transaction_Amount"].sort_values(ascending=False).reset_index(drop=True)
    cumsum  = fraud_amounts.cumsum()
    total   = fraud_amounts.sum()
    idx_80  = int((cumsum >= total * 0.80).idxmax())

    fig4 = go.Figure()
    fig4.add_trace(go.Bar(x=list(range(len(fraud_amounts))), y=fraud_amounts.values,
                          name="Fraud Amount", marker_color=CLR["danger"], opacity=0.7))
    fig4.add_trace(go.Scatter(x=list(range(len(fraud_amounts))),
                              y=(cumsum/total*100).values,
                              name="Cumulative %", yaxis="y2",
                              line=dict(color=CLR["dark"], width=2.5)))
    fig4.add_vline(x=idx_80, line_dash="dash", line_color=CLR["warning"],
                   annotation_text=f"80% fraud value at #{idx_80}")
    fig4.update_layout(
        title="Pareto — Fraudulent Transaction Amounts",
        xaxis_title="Transaction (ranked by amount)",
        yaxis=dict(title="Amount ($)"),
        yaxis2=dict(title="Cumulative %", overlaying="y", side="right",
                    range=[0,105], ticksuffix="%"),
        height=420, hovermode="x unified", legend=dict(x=0.01,y=0.99))
    st.plotly_chart(fig4, use_container_width=True)

    insight(f"Top {idx_80} fraudulent transactions account for 80% of total fraud value — Pareto confirmed.")
    warn("Fraudsters sometimes test with small amounts before large fraud — amount alone is not sufficient.")

# ════════════════════════════════════════════════════════════
# TAB 4 — TRANSACTION PATTERNS ★
# ════════════════════════════════════════════════════════════
with tabs[3]:
    sec("🔄 Tab 4 — Transaction Patterns ★")
    info("Velocity, previous fraud history, and account age are the strongest behavioral signals.")

    col1, col2 = st.columns(2)
    with col1:
        sec("⚡ Velocity — Fraud vs Legit")
        fig, ax = plt.subplots(figsize=(8,4))
        ax.hist(df_legit["Number_of_Transactions_Last_24H"], bins=30, alpha=0.6,
                color=CLR["primary"], label="Legitimate", density=True)
        ax.hist(df_fraud["Number_of_Transactions_Last_24H"], bins=30, alpha=0.7,
                color=CLR["danger"], label="Fraudulent", density=True)
        ax.set_xlabel("Transactions in Last 24H"); ax.set_ylabel("Density")
        ax.set_title("Velocity: Fraud vs Legitimate"); ax.legend()
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col2:
        sec("📜 Fraud Rate by Previous Fraud Count")
        prev = df.groupby("Previous_Fraudulent_Transactions")[TARGET].mean().reset_index()
        prev.columns = ["Prev_Fraud","Fraud_Rate"]
        prev["Fraud_Rate"] *= 100
        fig2, ax2 = plt.subplots(figsize=(8,4))
        ax2.bar(prev["Prev_Fraud"], prev["Fraud_Rate"],
                color=CLR["danger"], alpha=0.8, edgecolor="white")
        ax2.axhline(fraud_rate, color=CLR["primary"], lw=2, ls="--",
                    label=f"Avg {fraud_rate:.1f}%")
        ax2.set_xlabel("Previous Fraudulent Transactions")
        ax2.set_ylabel("Fraud Rate (%)")
        ax2.set_title("Fraud Rate by History"); ax2.legend()
        plt.tight_layout(); st.pyplot(fig2); plt.close()

    st.markdown("---")
    col3, col4 = st.columns(2)
    with col3:
        sec("🗓 Account Age — Fraud vs Legit")
        fig3, ax3 = plt.subplots(figsize=(8,4))
        ax3.hist(df_legit["Account_Age"], bins=40, alpha=0.6,
                 color=CLR["primary"], label="Legitimate", density=True)
        ax3.hist(df_fraud["Account_Age"], bins=40, alpha=0.7,
                 color=CLR["danger"], label="Fraudulent", density=True)
        ax3.set_xlabel("Account Age (days)"); ax3.set_ylabel("Density")
        ax3.set_title("Account Age: Fraud vs Legitimate"); ax3.legend()
        plt.tight_layout(); st.pyplot(fig3); plt.close()

    with col4:
        sec("📊 Fraud vs Legit — Metric Comparison")
        compare = pd.DataFrame({
            "Metric": ["Avg Amount ($)","Avg Velocity (24H)",
                       "Avg Account Age","Avg Prev Fraud"],
            "Legitimate": [df_legit["Transaction_Amount"].mean(),
                           df_legit["Number_of_Transactions_Last_24H"].mean(),
                           df_legit["Account_Age"].mean(),
                           df_legit["Previous_Fraudulent_Transactions"].mean()],
            "Fraudulent":  [df_fraud["Transaction_Amount"].mean(),
                            df_fraud["Number_of_Transactions_Last_24H"].mean(),
                            df_fraud["Account_Age"].mean(),
                            df_fraud["Previous_Fraudulent_Transactions"].mean()],
        }).round(2)
        compare["Ratio F/L"] = (compare["Fraudulent"]/compare["Legitimate"]).round(3)
        st.dataframe(compare, use_container_width=True)
        fig4 = px.bar(compare, x="Metric", y="Ratio F/L",
                      color="Ratio F/L",
                      color_continuous_scale=["#2e7d32","#e65100","#c62828"],
                      title="Fraud/Legit Ratio (>1 = fraud is higher)")
        fig4.add_hline(y=1.0, line_dash="dash", line_color="black",
                       annotation_text="Equal line")
        fig4.update_layout(height=300)
        st.plotly_chart(fig4, use_container_width=True)

    insight("Previous_Fraudulent_Transactions is the strongest single predictor — past behavior predicts future fraud.")
    insight("New accounts (low Account_Age) show disproportionately higher fraud — a classic pattern.")
    fraud_alert("High velocity + new account + night time = highest risk combination.")

# ════════════════════════════════════════════════════════════
# TAB 5 — TIME ANALYSIS ★
# ════════════════════════════════════════════════════════════
with tabs[4]:
    sec("⏰ Tab 5 — Time Analysis ★")
    info("Fraud patterns change by hour. Night-time (10 PM–6 AM) shows elevated fraud when oversight is minimal.")

    if "Hour" in df.columns:
        hourly = df.groupby("Hour")[TARGET].agg(Total="count",Fraud="sum").reset_index()
        hourly["Fraud_Rate%"] = (hourly["Fraud"]/hourly["Total"]*100).round(2)

        sec("📈 Fraud Rate by Hour of Day")
        fig = go.Figure()
        fig.add_trace(go.Bar(x=hourly["Hour"], y=hourly["Total"],
                             name="Total Transactions", marker_color=CLR["light"],
                             opacity=0.6, yaxis="y"))
        fig.add_trace(go.Scatter(x=hourly["Hour"], y=hourly["Fraud_Rate%"],
                                 name="Fraud Rate %", yaxis="y2",
                                 line=dict(color=CLR["danger"], width=3),
                                 marker=dict(size=7)))
        fig.add_vrect(x0=21.5, x1=23.5, fillcolor="red", opacity=0.07,
                      annotation_text="Late Night")
        fig.add_vrect(x0=-0.5, x1=5.5, fillcolor="red", opacity=0.07,
                      annotation_text="Early AM")
        fig.update_layout(
            title="Transactions vs Fraud Rate by Hour",
            xaxis_title="Hour of Day",
            yaxis=dict(title="Transaction Count"),
            yaxis2=dict(title="Fraud Rate %", overlaying="y", side="right",
                        ticksuffix="%"),
            height=420, hovermode="x unified", legend=dict(x=0.01,y=0.99))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            sec("🌙 Night vs Day")
            if "is_night" in df.columns:
                nc = df.groupby("is_night")[TARGET].agg(Total="count",Fraud="sum").reset_index()
                nc["Label"]      = nc["is_night"].map({0:"Day (6AM–10PM)",1:"Night (10PM–6AM)"})
                nc["Fraud_Rate%"] = (nc["Fraud"]/nc["Total"]*100).round(2)
                fig2 = px.bar(nc, x="Label", y="Fraud_Rate%", color="Label",
                              color_discrete_map={"Day (6AM–10PM)":CLR["amber"],
                                                  "Night (10PM–6AM)":CLR["dark"]},
                              title="Fraud Rate: Day vs Night",
                              text=nc["Fraud_Rate%"].apply(lambda x: f"{x:.2f}%"))
                fig2.update_traces(textposition="outside")
                fig2.update_layout(height=350, showlegend=False)
                st.plotly_chart(fig2, use_container_width=True)

        with col2:
            sec("💼 Business Hours vs Off-Hours")
            if "is_business_hours" in df.columns:
                bc = df.groupby("is_business_hours")[TARGET].agg(Total="count",Fraud="sum").reset_index()
                bc["Label"]      = bc["is_business_hours"].map(
                    {0:"Off-Hours",1:"Business Hours (9AM–5PM)"})
                bc["Fraud_Rate%"] = (bc["Fraud"]/bc["Total"]*100).round(2)
                fig3 = px.bar(bc, x="Label", y="Fraud_Rate%", color="Label",
                              color_discrete_map={"Business Hours (9AM–5PM)":CLR["success"],
                                                  "Off-Hours":CLR["warning"]},
                              title="Fraud Rate: Business vs Off Hours",
                              text=bc["Fraud_Rate%"].apply(lambda x: f"{x:.2f}%"))
                fig3.update_traces(textposition="outside")
                fig3.update_layout(height=350, showlegend=False)
                st.plotly_chart(fig3, use_container_width=True)

    insight("Night-time fraud is elevated — fraudsters exploit reduced monitoring windows.")
    warn("Time alone is a weak signal — combine with velocity, amount, and device for best detection.")

# ════════════════════════════════════════════════════════════
# TAB 6 — DEVICE & LOCATION ★
# ════════════════════════════════════════════════════════════
with tabs[5]:
    sec("📍 Tab 6 — Device & Location ★")

    col1, col2 = st.columns(2)
    with col1:
        sec("📱 Fraud Rate by Device")
        dev = df.groupby("Device_Used")[TARGET].agg(Total="count",Fraud="sum").reset_index()
        dev["Fraud_Rate%"] = (dev["Fraud"]/dev["Total"]*100).round(2)
        dev = dev.sort_values("Fraud_Rate%", ascending=False)
        fig = px.bar(dev, x="Device_Used", y="Fraud_Rate%",
                     color="Fraud_Rate%",
                     color_continuous_scale=["#2e7d32","#e65100","#c62828"],
                     title="Fraud Rate % by Device",
                     text=dev["Fraud_Rate%"].apply(lambda x: f"{x:.1f}%"))
        fig.add_hline(y=fraud_rate, line_dash="dash", line_color="blue",
                      annotation_text=f"Avg {fraud_rate:.1f}%")
        fig.update_traces(textposition="outside")
        fig.update_layout(height=380, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        sec("🗺 Fraud Rate by Location")
        loc = df.groupby("Location")[TARGET].agg(Total="count",Fraud="sum").reset_index()
        loc["Fraud_Rate%"] = (loc["Fraud"]/loc["Total"]*100).round(2)
        loc = loc.sort_values("Fraud_Rate%", ascending=False)
        fig2 = px.bar(loc, x="Location", y="Fraud_Rate%",
                      color="Fraud_Rate%",
                      color_continuous_scale=["#2e7d32","#e65100","#c62828"],
                      title="Fraud Rate % by City",
                      text=loc["Fraud_Rate%"].apply(lambda x: f"{x:.1f}%"))
        fig2.add_hline(y=fraud_rate, line_dash="dash", line_color="blue",
                       annotation_text=f"Avg {fraud_rate:.1f}%")
        fig2.update_traces(textposition="outside")
        fig2.update_layout(height=380, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    col3, col4 = st.columns(2)
    with col3:
        dev_vol = df["Device_Used"].value_counts().reset_index()
        dev_vol.columns = ["Device","Count"]
        fig3 = px.pie(dev_vol, names="Device", values="Count",
                      color_discrete_sequence=[CLR["primary"],CLR["teal"],
                                               CLR["warning"],CLR["purple"],CLR["grey"]],
                      title="Transaction Volume by Device", hole=0.4)
        fig3.update_layout(height=350)
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        loc_vol = df["Location"].value_counts().reset_index()
        loc_vol.columns = ["Location","Count"]
        fig4 = px.bar(loc_vol, x="Location", y="Count",
                      color="Count", color_continuous_scale="Blues",
                      title="Transaction Volume by City")
        fig4.update_layout(height=350)
        st.plotly_chart(fig4, use_container_width=True)

    insight("Unknown device = unrecognized device → strong fraud signal → trigger step-up authentication.")
    warn("Location alone is weak — VPN and device spoofing can mask true origin.")

# ════════════════════════════════════════════════════════════
# TAB 7 — FEATURE ENGINEERING
# ════════════════════════════════════════════════════════════
with tabs[6]:
    sec("⚙️ Tab 7 — Feature Engineering")

    col1, col2 = st.columns(2)
    with col1:
        sec("📋 All Engineered Features")
        fe = pd.DataFrame({
            "Feature":  ["Hour","is_night","is_business_hours","high_velocity",
                         "new_account","Amount_Category",
                         "Transaction_Type_enc","Device_Used_enc",
                         "Location_enc","Payment_Method_enc"],
            "Source":   ["Time_of_Transaction renamed",
                         "Hour ∈ {22,23,0,1,2,3,4,5}",
                         "Hour ∈ [9–17]",
                         "Transactions_24H > Q75",
                         "Account_Age < Q25",
                         "Amount binned 5 ranges",
                         "LabelEncoder","LabelEncoder",
                         "LabelEncoder","LabelEncoder"],
            "Reason":   ["Hour enables time-pattern detection",
                         "Night fraud elevated — reduced oversight",
                         "Business hours = lower fraud risk",
                         "Rapid transactions = card testing",
                         "New accounts = higher risk",
                         "Capture non-linear amount effect",
                         "ML-ready","ML-ready","ML-ready","ML-ready"],
        })
        st.dataframe(fe, use_container_width=True)

    with col2:
        sec("⚡ High Velocity Flag — Fraud Rate")
        if "high_velocity" in df.columns:
            vc = df.groupby("high_velocity")[TARGET].agg(Total="count",Fraud="sum").reset_index()
            vc["Label"]      = vc["high_velocity"].map({0:"Normal",1:"High Velocity"})
            vc["Fraud_Rate%"] = (vc["Fraud"]/vc["Total"]*100).round(2)
            fig = px.bar(vc, x="Label", y="Fraud_Rate%", color="Label",
                         color_discrete_map={"Normal":CLR["success"],
                                             "High Velocity":CLR["danger"]},
                         title="Fraud Rate: Normal vs High Velocity",
                         text=vc["Fraud_Rate%"].apply(lambda x: f"{x:.2f}%"))
            fig.update_traces(textposition="outside")
            fig.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    sec("🆕 New Account Flag — Fraud Rate")
    if "new_account" in df.columns:
        na = df.groupby("new_account")[TARGET].agg(Total="count",Fraud="sum").reset_index()
        na["Label"]      = na["new_account"].map({0:"Established",1:"New Account"})
        na["Fraud_Rate%"] = (na["Fraud"]/na["Total"]*100).round(2)
        col3, col4 = st.columns(2)
        with col3:
            st.dataframe(na[["Label","Total","Fraud","Fraud_Rate%"]], use_container_width=True)
        with col4:
            fig2 = px.bar(na, x="Label", y="Fraud_Rate%", color="Label",
                          color_discrete_map={"Established":CLR["success"],
                                              "New Account":CLR["danger"]},
                          title="Fraud Rate: New vs Established Accounts",
                          text=na["Fraud_Rate%"].apply(lambda x: f"{x:.2f}%"))
            fig2.update_traces(textposition="outside")
            fig2.update_layout(height=320, showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)

    insight("High velocity + new account = highest risk profile combination.")
    info("_enc columns used for ML only — original categorical columns kept for EDA display.")

# ════════════════════════════════════════════════════════════
# TAB 8 — CORRELATION
# ════════════════════════════════════════════════════════════
with tabs[7]:
    sec("🔥 Tab 8 — Correlation Analysis")

    corr_cols = [c for c in NUM_COLS + ENG_COLS + ENC_COLS + [TARGET] if c in df.columns]
    corr = df[corr_cols].corr()

    fig, ax = plt.subplots(figsize=(13,9))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlBu_r",
                vmin=-1, vmax=1, ax=ax, linewidths=0.5, annot_kws={"size":8})
    ax.set_title("Full Correlation Matrix — Fraud Detection Features",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("---")
    sec("🎯 Top Correlations with Fraudulent")
    tgt_corr = corr[TARGET].drop(TARGET).sort_values(key=abs, ascending=False)
    fig2, ax2 = plt.subplots(figsize=(10,5))
    colors_bar = [CLR["danger"] if v > 0 else CLR["success"] for v in tgt_corr.values]
    ax2.barh(tgt_corr.index, tgt_corr.values, color=colors_bar)
    ax2.axvline(0, color="black", lw=0.8)
    ax2.set_xlabel("Pearson r with Fraudulent")
    ax2.set_title("Feature Correlation with Fraud Target", fontsize=12, fontweight="bold")
    for i, (idx, val) in enumerate(tgt_corr.items()):
        ax2.text(val + 0.003 if val >= 0 else val - 0.003, i,
                 f"{val:.3f}", va="center",
                 ha="left" if val >= 0 else "right", fontsize=9)
    plt.tight_layout(); st.pyplot(fig2); plt.close()

    insight("Previous_Fraudulent_Transactions has the strongest correlation with fraud target.")
    warn("Low Pearson r doesn't mean useless — tree models capture non-linear relationships Pearson misses.")
    info("For imbalanced data, feature importance from RF/GB is more reliable than Pearson r.")

# ════════════════════════════════════════════════════════════
# TAB 9 — ANOMALY DETECTION ★
# ════════════════════════════════════════════════════════════
with tabs[8]:
    sec("🚨 Tab 9 — Anomaly Detection ★")
    info("Extreme transaction amounts and behavioral outliers — do they concentrate fraud?")

    z = np.abs(zscore(df["Transaction_Amount"].dropna()))
    outlier_mask = z > 3
    n_out = outlier_mask.sum()
    fraud_in_out = df.loc[df["Transaction_Amount"].dropna().index[outlier_mask], TARGET].sum()

    c1,c2,c3 = st.columns(3)
    c1.metric("Outliers (|Z|>3)",      f"{n_out:,}")
    c2.metric("Fraud in Outliers",      f"{fraud_in_out:,}")
    c3.metric("Fraud Rate in Outliers",
              f"{fraud_in_out/n_out*100:.1f}%" if n_out > 0 else "N/A")

    fig, ax = plt.subplots(figsize=(12,4))
    ax.scatter(range(len(z)), z,
               c=[CLR["danger"] if o else CLR["primary"] for o in outlier_mask],
               alpha=0.4, s=5)
    ax.axhline(3, color=CLR["danger"], lw=2, ls="--", label="Z=3 threshold")
    ax.set_xlabel("Transaction Index"); ax.set_ylabel("|Z-Score|")
    ax.set_title("Z-Score Anomaly Detection — Transaction Amounts")
    ax.legend(); plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("---")
    sec("📊 IQR Outliers — All Numeric Features")
    iqr_rows = []
    for col in NUM_COLS:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        out = df[(df[col] < Q1-1.5*IQR) | (df[col] > Q3+1.5*IQR)]
        fr  = out[TARGET].sum() if len(out) > 0 else 0
        iqr_rows.append({
            "Feature": col, "Outliers": len(out),
            "Outlier%": round(len(out)/len(df)*100,2),
            "Fraud in Outliers": fr,
            "Fraud Rate%": round(fr/len(out)*100,2) if len(out)>0 else 0
        })
    st.dataframe(pd.DataFrame(iqr_rows).sort_values("Outliers",ascending=False),
                 use_container_width=True)

    st.markdown("---")
    sec("📈 Top 200 High-Amount Transactions — Fraud Highlighted")
    top200 = df.nlargest(200,"Transaction_Amount")[
        ["Transaction_Amount","Number_of_Transactions_Last_24H","Account_Age",TARGET]]
    fig2 = px.scatter(top200, x="Transaction_Amount",
                      y="Number_of_Transactions_Last_24H",
                      color=TARGET, size="Transaction_Amount",
                      color_discrete_map={0:CLR["primary"],1:CLR["danger"]},
                      hover_data=["Account_Age"],
                      title="Top 200 Highest-Amount Transactions")
    fig2.update_layout(height=420)
    st.plotly_chart(fig2, use_container_width=True)

    fraud_alert("Outlier transactions (|Z|>3) show disproportionately high fraud rate — extreme amounts are a strong signal.")
    warn("Not all outliers are fraud — ML models needed to combine multiple signals.")

# ════════════════════════════════════════════════════════════
# TAB 10 — A/B TEST ★
# ════════════════════════════════════════════════════════════
with tabs[9]:
    sec("🧪 Tab 10 — A/B Test: High Velocity vs Normal ★")
    info("Hypothesis: High-velocity users transact at significantly different amounts than normal users.")

    if "high_velocity" in df.columns:
        gA = df[df["high_velocity"]==0]["Transaction_Amount"]
        gB = df[df["high_velocity"]==1]["Transaction_Amount"]

        t_stat, p_val = stats.ttest_ind(gA, gB, equal_var=False)
        pooled        = np.sqrt((gA.std()**2 + gB.std()**2) / 2)
        cohens_d      = (gB.mean() - gA.mean()) / pooled
        diff          = gB.mean() - gA.mean()
        se            = np.sqrt(gA.var()/len(gA) + gB.var()/len(gB))
        ci_lo, ci_hi  = diff - 1.96*se, diff + 1.96*se

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Normal — Avg Amount",  f"${gA.mean():,.2f}")
        c2.metric("High Vel — Avg Amount",f"${gB.mean():,.2f}")
        c3.metric("p-value",              f"{p_val:.4f}")
        c4.metric("Cohen's d",            f"{cohens_d:.3f}")

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(8,4))
            ax.hist(gA, bins=40, alpha=0.6, color=CLR["primary"],
                    label=f"Normal (n={len(gA):,})", density=True)
            ax.hist(gB, bins=40, alpha=0.6, color=CLR["danger"],
                    label=f"High Vel (n={len(gB):,})", density=True)
            ax.axvline(gA.mean(), color=CLR["primary"], lw=2, ls="--")
            ax.axvline(gB.mean(), color=CLR["danger"],  lw=2, ls="--")
            ax.set_xlabel("Amount ($)"); ax.set_ylabel("Density")
            ax.set_title("Normal vs High Velocity — Amount"); ax.legend()
            plt.tight_layout(); st.pyplot(fig); plt.close()

        with col2:
            fig2, ax2 = plt.subplots(figsize=(6,4))
            bp = ax2.boxplot([gA.dropna(), gB.dropna()], patch_artist=True,
                             labels=["Normal","High Velocity"])
            bp["boxes"][0].set_facecolor(CLR["light"])
            bp["boxes"][1].set_facecolor("#fce4ec")
            for m in bp["medians"]: m.set_color(CLR["danger"]); m.set_linewidth(2)
            ax2.set_ylabel("Amount ($)"); ax2.set_title("Box Plot: Velocity Groups")
            plt.tight_layout(); st.pyplot(fig2); plt.close()

        st.markdown("---")
        sec("📋 Test Results")
        res = pd.DataFrame({
            "Metric": ["Test","H₀","H₁","t-statistic","p-value",
                       "Significant (α=0.05)","Cohen's d","Effect Size",
                       "95% CI","Decision"],
            "Result": [
                "Welch T-Test (unequal variance)",
                "High velocity amount = Normal amount",
                "High velocity amount ≠ Normal amount",
                f"{t_stat:.4f}", f"{p_val:.6f}",
                "✅ YES" if p_val < 0.05 else "❌ NO",
                f"{cohens_d:.4f}",
                "Large" if abs(cohens_d)>0.8 else "Medium" if abs(cohens_d)>0.5 else "Small",
                f"[{ci_lo:.2f}, {ci_hi:.2f}]",
                "✅ REJECT H₀" if p_val < 0.05 else "❌ FAIL to reject H₀"
            ]
        })
        st.dataframe(res, use_container_width=True)

        if p_val < 0.05:
            diff_pct = (gB.mean()-gA.mean())/gA.mean()*100
            insight(f"High velocity users transact at avg ${gB.mean():,.2f} vs ${gA.mean():,.2f} "
                    f"— {diff_pct:.1f}% difference. Statistically significant.")
            fraud_alert("Flag transactions from users exceeding 75th velocity percentile for additional verification.")

# ════════════════════════════════════════════════════════════
# TAB 11 — MISSING VALUES
# ════════════════════════════════════════════════════════════
with tabs[10]:
    sec("❓ Tab 11 — Missing Values & Imputation")

    null_counts = df.isnull().sum()
    null_pct    = (null_counts/len(df)*100).round(2)
    miss = pd.DataFrame({
        "Column":  null_counts.index,
        "Missing": null_counts.values,
        "Percent": null_pct.values,
    }).sort_values("Missing", ascending=False).reset_index(drop=True)
    miss_only = miss[miss["Missing"] > 0]

    if miss_only.empty:
        insight("No missing values — dataset is perfectly clean after preprocessing.")
    else:
        st.dataframe(miss_only, use_container_width=True)
        fig = px.bar(miss_only, x="Column", y="Percent",
                     color="Percent",
                     color_continuous_scale=["#2e7d32","#e65100","#c62828"],
                     title="Missing % per Column")
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    sec("📋 Imputation Strategy Applied")
    strat = pd.DataFrame({
        "Column":   ["Transaction_Amount","Time_of_Transaction (→ Hour)",
                     "Device_Used","Location","Payment_Method"],
        "Missing":  ["~5%","~5%","~5%","~5%","~5%"],
        "Strategy": ["Median","Median → int",
                     "Fill 'Unknown'","Fill 'Unknown'",
                     "Fill 'Unknown' + replace 'Invalid Method'"],
        "Reason":   ["Numeric — median robust to outliers",
                     "Already float hour — median preserves distribution",
                     "Categorical — Unknown is a valid, informative value",
                     "Categorical — Unknown location is itself a signal",
                     "Invalid Method = effectively unknown — unified category"]
    })
    st.dataframe(strat, use_container_width=True)

    insight("~5% missing per column — low enough that simple imputation is safe.")
    warn("'Unknown' device/location/payment is itself a fraud signal — not just noise.")

# ════════════════════════════════════════════════════════════
# TAB 12 — MULTICOLLINEARITY
# ════════════════════════════════════════════════════════════
with tabs[11]:
    sec("🔁 Tab 12 — Multicollinearity / VIF")
    info("VIF > 10 = severe multicollinearity → matters for linear models, not tree-based.")

    vif_cols = [c for c in NUM_COLS + ENG_COLS if c in df.columns]
    vif_data = df[vif_cols].dropna()

    try:
        vif_df = pd.DataFrame({
            "Feature": vif_cols,
            "VIF": [round(variance_inflation_factor(vif_data.values, i), 2)
                    for i in range(len(vif_cols))]
        }).sort_values("VIF", ascending=False)
        vif_df["Risk"] = vif_df["VIF"].apply(
            lambda v: "🔴 High" if v>10 else "🟡 Medium" if v>5 else "🟢 Low")

        col1, col2 = st.columns([1,1.5])
        with col1:
            st.dataframe(vif_df, use_container_width=True)
        with col2:
            fig, ax = plt.subplots(figsize=(7,5))
            colors_vif = [CLR["danger"] if v>10 else CLR["warning"] if v>5
                          else CLR["success"] for v in vif_df["VIF"]]
            ax.barh(vif_df["Feature"], vif_df["VIF"], color=colors_vif)
            ax.axvline(10, color=CLR["danger"],  lw=2, ls="--", label="VIF=10")
            ax.axvline(5,  color=CLR["warning"], lw=1.5, ls=":",  label="VIF=5")
            ax.set_xlabel("VIF Score")
            ax.set_title("VIF — Multicollinearity Check", fontsize=12, fontweight="bold")
            ax.legend(); plt.tight_layout(); st.pyplot(fig); plt.close()
    except Exception as e:
        warn(f"VIF error: {e}")

    warn("High VIF features don't need dropping for RF/GB — only for Logistic Regression.")
    insight("Engineered flags (is_night, high_velocity) derived from existing features — moderate VIF expected.")

# ════════════════════════════════════════════════════════════
# TAB 13 — INSIGHTS & REPORT
# ════════════════════════════════════════════════════════════
with tabs[12]:
    sec("💡 Tab 13 — Insights & Recommendations")

    st.markdown(f"### 🔍 Fraud Detection — Final Analysis Report")
    st.markdown(f"**{len(df):,} transactions · {len(df_fraud):,} fraudulent ({fraud_rate:.1f}%) · Kaggle Dataset · M3**")
    st.markdown("---")

    sec("1️⃣ Class Imbalance — Critical")
    fraud_alert(f"Only {fraud_rate:.1f}% fraud — predicting all 'Legitimate' gives 95% accuracy but catches 0 fraud.")
    insight("class_weight='balanced' applied to all classifiers. Evaluate with F1, Precision, Recall, ROC-AUC.")

    st.markdown("---")
    sec("2️⃣ Top Fraud Signals")
    insight("Previous_Fraudulent_Transactions — strongest predictor. Past fraud = future fraud risk.")
    insight("High velocity (Number_of_Transactions_Last_24H > Q75) — card testing pattern.")
    insight("New accounts (Account_Age < Q25) — fraudsters open fresh accounts to bypass history checks.")
    insight("Night-time transactions (10 PM – 6 AM) — elevated fraud when human oversight is minimal.")
    insight("Unknown device — unrecognized device = high-risk authentication trigger.")
    insight("Unknown / Invalid payment method — data anomaly itself is a signal.")

    st.markdown("---")
    sec("3️⃣ Amount Patterns")
    insight("Pareto confirmed: top fraudulent transactions drive 80% of fraud value.")
    warn("Fraudsters also test with small amounts before large fraud — multi-threshold rules needed.")

    st.markdown("---")
    sec("4️⃣ Statistical Evidence")
    insight("A/B Test: high-velocity users show significantly different transaction patterns (p < 0.05).")
    insight("Z-score outliers (|Z|>3) in amount show disproportionate fraud concentration.")
    warn("Pearson r values are low — fraud has non-linear relationships best captured by RF/GB models.")

    st.markdown("---")
    sec("5️⃣ Recommendations")
    recs = [
        ("🚨 Real-Time ML Scoring",   "Deploy model to score every transaction — flag high-risk for review."),
        ("⚡ Velocity Rule",           "Block/flag users exceeding 75th percentile transactions per 24H."),
        ("🆕 New Account Limits",      "Apply transaction caps on accounts below 25th percentile age."),
        ("🌙 Night Monitoring",        "Increase automated scrutiny between 10 PM – 6 AM."),
        ("📜 History Weighting",       "Weight Previous_Fraudulent_Transactions heavily in pre-filter rules."),
        ("📱 Device Fingerprinting",   "Unknown device = automatic OTP / biometric step-up authentication."),
        ("💳 Payment Method Checks",   "Flag 'Unknown' / 'Invalid Method' transactions for manual review."),
    ]
    for title, text in recs:
        st.markdown(
            f'<div class="fraud-box"><p><b>{title}:</b> {text}</p></div>',
            unsafe_allow_html=True)

    st.markdown("---")
    report_txt = f"""FRAUD DETECTION — FINAL REPORT
M3 · Kaggle Dataset · {len(df):,} Transactions
Fraudulent: {len(df_fraud):,} ({fraud_rate:.1f}%) | Legitimate: {len(df_legit):,}

CLASS IMBALANCE:
- Fraud rate {fraud_rate:.1f}% — class_weight='balanced' mandatory
- Evaluate: F1, Precision, Recall, ROC-AUC (NOT accuracy)

TOP SIGNALS:
1. Previous_Fraudulent_Transactions — strongest predictor
2. High velocity (Transactions last 24H) — card testing
3. New account (low Account_Age) — fresh account risk
4. Night-time (10PM–6AM) — reduced oversight
5. Unknown device — unrecognized device flag
6. Unknown/Invalid payment method — data anomaly signal

RECOMMENDATIONS:
- Real-time ML scoring on every transaction
- Velocity flag > 75th percentile
- New account transaction limits
- Night-time automated scrutiny
- Unknown device = step-up auth
- Flag Unknown/Invalid payment methods
"""
    col1, col2 = st.columns(2)
    with col1:
        st.download_button("📥 Download Report (.txt)", report_txt,
                           file_name="FraudDetection_Report_M3.txt",
                           mime="text/plain", use_container_width=True)
    with col2:
        st.download_button("📥 Download Clean Data (.csv)",
                           df.to_csv(index=False),
                           file_name="fraud_clean_M3.csv",
                           mime="text/csv", use_container_width=True)
