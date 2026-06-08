"""
Repo_7_Fraud_Detection — ML_Models.py  (5 Tabs)
Author : Mohamed · M3
Dataset: Kaggle Fraud Detection · 51,000 transactions
Regression  → Transaction_Amount
Classification → Fraudulent  (class_weight='balanced' — MANDATORY)
"""
# streamlit run "G:\FINAL_PROJECTS\P7_Fraud_Detection\pages\ML_Models.py"

import os, pathlib, warnings, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import psutil

# ── sklearn ───────────────────────────────────────────────────
from sklearn.model_selection   import train_test_split
from sklearn.preprocessing     import StandardScaler, LabelEncoder
from sklearn.pipeline          import Pipeline
from sklearn.metrics           import (
    r2_score, mean_absolute_error, mean_squared_error,
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve
)
# Regression models
from sklearn.linear_model      import LinearRegression, Ridge, Lasso
from sklearn.tree              import DecisionTreeRegressor
from sklearn.ensemble          import RandomForestRegressor, GradientBoostingRegressor
# Classification models
from sklearn.linear_model      import LogisticRegression
from sklearn.tree              import DecisionTreeClassifier
from sklearn.ensemble          import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm               import LinearSVC
from sklearn.calibration       import CalibratedClassifierCV
from sklearn.neighbors         import KNeighborsClassifier

warnings.filterwarnings("ignore")
S = st.session_state

# ── PAGE CONFIG ──────────────────────────────────────────────
st.set_page_config(page_title="ML Models · Fraud Detection · M3",
                   page_icon="🤖", layout="wide")

# ── PATHS ────────────────────────────────────────────────────
LOGO = pathlib.Path(__file__).parent.parent / "M3_logo.png"
DATA = pathlib.Path(__file__).parent.parent / "data" / "fraud_clean.csv"

# ── SIDEBAR ──────────────────────────────────────────────────
with st.sidebar:
    if LOGO.exists():
        st.image(str(LOGO), width=70)
    st.markdown("### 🤖 ML Models")
    st.markdown("Fraud Detection · 5 Tabs")
    st.divider()
    st.markdown("### 📂 Dataset")
    _uploaded = st.file_uploader("Upload Clean CSV", type=["csv"],
                                  help="Upload fraud_clean.csv from Jupyter.",
                                  key="ml_upload")
    if _uploaded is not None:
        st.success(f"✅ Using: {_uploaded.name}")
    else:
        st.info("Using default: fraud_clean.csv")
    st.divider()
    st.markdown("### ⚙️ Training Options")
    test_size    = st.slider("Test Split %", 10, 40, 20, 5, key="ts") / 100
    use_parallel = st.checkbox("Parallel Training (n_jobs=-1)", value=True, key="par")
    n_jobs       = -1 if use_parallel else 1
    st.markdown("### 🎯 Imbalance Strategy")
    st.error("class_weight='balanced'\napplied to ALL classifiers")

# ── PALETTE ──────────────────────────────────────────────────
CLR = {"primary":"#1565c0","success":"#2e7d32","warning":"#e65100",
       "danger":"#c62828","teal":"#00695c","accent":"#00695c",
       "secondary":"#455a64","light":"#e3f2fd","dark":"#1a237e",
       "purple":"#6a1b9a","amber":"#f57f17","pink":"#ad1457",
       "indigo":"#283593","cyan":"#00838f","lime":"#558b2f",
       "brown":"#4e342e","grey":"#546e7a","white":"#ffffff","black":"#212121"}

MODEL_COLORS = [CLR["primary"],CLR["success"],CLR["warning"],
                CLR["danger"],CLR["teal"],CLR["purple"]]

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
div[data-testid="metric-container"]{background:#e3f2fd;border-left:4px solid #1565c0;border-radius:6px;padding:10px 14px;}
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

def get_cpu_info(use_parallel: bool, n_jobs: int = 1) -> dict:
    return {"total": os.cpu_count(),
            "used":  n_jobs if use_parallel else 1,
            "percent": psutil.cpu_percent(interval=0.3)}

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

_up = S.get("ml_upload", None)
if _up is not None:
    _bytes = _up.read(); _up.seek(0)
    df = load_data(file_bytes=_bytes)
else:
    df = load_data()

if df.empty:
    st.warning("⚠️ No data found. Upload fraud_clean.csv or run P7_clean_data.py first.")
    st.stop()

S["df_work"] = df

# ════════════════════════════════════════════════════════════
# FEATURE PREP
# ════════════════════════════════════════════════════════════
ENC_COLS = [c for c in df.columns if c.endswith("_enc")]
NUM_FEATS = [c for c in ["Previous_Fraudulent_Transactions","Account_Age",
                          "Number_of_Transactions_Last_24H","Hour",
                          "is_night","is_business_hours","high_velocity","new_account"]
             if c in df.columns]
ALL_FEATS = NUM_FEATS + ENC_COLS

REG_TARGET = "Transaction_Amount"
CLF_TARGET = "Fraudulent"

# Drop rows where targets are null
df_ml = df[ALL_FEATS + [REG_TARGET, CLF_TARGET]].dropna().copy()

X      = df_ml[ALL_FEATS]
y_reg  = df_ml[REG_TARGET]
y_clf  = df_ml[CLF_TARGET]

X_train_r, X_test_r, yr_train, yr_test = train_test_split(
    X, y_reg, test_size=test_size, random_state=42)
X_train_c, X_test_c, yc_train, yc_test = train_test_split(
    X, y_clf, test_size=test_size, random_state=42, stratify=y_clf)

scaler    = StandardScaler()
Xtr_r_sc  = scaler.fit_transform(X_train_r)
Xte_r_sc  = scaler.transform(X_test_r)
Xtr_c_sc  = scaler.fit_transform(X_train_c)
Xte_c_sc  = scaler.transform(X_test_c)

# ── Model definitions ─────────────────────────────────────────
REG_MODELS = {
    "Linear Regression":      LinearRegression(),
    "Ridge":                  Ridge(alpha=1.0),
    "Lasso":                  Lasso(alpha=0.1, max_iter=5000),
    "Decision Tree":          DecisionTreeRegressor(max_depth=8, random_state=42),
    "Random Forest":          RandomForestRegressor(n_estimators=100, n_jobs=n_jobs, random_state=42),
    "Gradient Boosting":      GradientBoostingRegressor(n_estimators=100, random_state=42),
}
CLF_MODELS = {
    "Logistic Regression":    LogisticRegression(max_iter=1000, class_weight="balanced",
                                                  n_jobs=n_jobs, random_state=42),
    "Decision Tree":          DecisionTreeClassifier(max_depth=8, class_weight="balanced",
                                                      random_state=42),
    "Random Forest":          RandomForestClassifier(n_estimators=100, class_weight="balanced",
                                                      n_jobs=n_jobs, random_state=42),
    "Gradient Boosting":      GradientBoostingClassifier(n_estimators=100, random_state=42),
    "SVM (Linear)":           CalibratedClassifierCV(
                                  LinearSVC(class_weight="balanced",
                                            max_iter=2000, random_state=42)),
    "KNN":                    KNeighborsClassifier(n_neighbors=7, n_jobs=n_jobs),
}

# ════════════════════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════════════════════
tabs = st.tabs([
    "1 · Model Training",
    "2 · Regression Results",
    "3 · Classification Results",
    "4 · Feature Importance",
    "5 · Predict",
])

# ════════════════════════════════════════════════════════════
# TAB 1 — MODEL TRAINING
# ════════════════════════════════════════════════════════════
with tabs[0]:
    sec("🚀 Tab 1 — Model Training")

    # Dataset summary
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Total Rows",       f"{len(df_ml):,}")
    c2.metric("Features Used",    f"{len(ALL_FEATS)}")
    c3.metric("Train Size",       f"{len(X_train_r):,}")
    c4.metric("Test Size",        f"{len(X_test_r):,}")
    c5.metric("Test Split",       f"{int(test_size*100)}%")

    fraud_rate = y_clf.mean() * 100
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        sec("🎯 Regression Target")
        st.markdown(f"**Target:** `{REG_TARGET}`")
        st.markdown(f"**Mean:** ${y_reg.mean():,.2f} · **Median:** ${y_reg.median():,.2f}")
        st.markdown(f"**Range:** ${y_reg.min():,.2f} – ${y_reg.max():,.2f}")
    with col2:
        sec("🎯 Classification Target")
        st.markdown(f"**Target:** `{CLF_TARGET}`")
        st.markdown(f"**Fraud Rate:** {fraud_rate:.2f}% → severe imbalance")
        st.markdown("**Strategy:** `class_weight='balanced'` on all classifiers")

    st.markdown("---")
    sec("📋 Features Used for ML")
    feat_df = pd.DataFrame({
        "Feature": ALL_FEATS,
        "Type": ["Numeric" if c in NUM_FEATS else "Encoded Categorical" for c in ALL_FEATS],
        "Description": [
            "Count of user's past fraudulent transactions",
            "Account age in days",
            "Transactions by this user in last 24 hours",
            "Hour of transaction (0–23)",
            "1 = Night-time transaction (10PM–6AM)",
            "1 = Business hours (9AM–5PM)",
            "1 = Velocity > 75th percentile",
            "1 = Account_Age < 25th percentile",
            "Transaction type encoded",
            "Device used encoded",
            "Location encoded",
            "Payment method encoded",
        ][:len(ALL_FEATS)]
    })
    st.dataframe(feat_df, use_container_width=True)

    st.markdown("---")
    cpu = get_cpu_info(use_parallel, n_jobs if use_parallel else 1)
    st.info(f"🖥 CPU: {cpu['total']} cores · Using: {cpu['used']} · Load: {cpu['percent']}%")
    fraud_alert(f"Fraud rate = {fraud_rate:.1f}% — evaluating by F1 + ROC-AUC, NOT accuracy.")

    # ── initialise storage lists if not present ───────────────
    if "reg_results" not in S: S["reg_results"] = []
    if "reg_models"  not in S: S["reg_models"]  = {}
    if "clf_results" not in S: S["clf_results"] = []
    if "clf_models"  not in S: S["clf_models"]  = {}
    S["X_test_r"]  = X_test_r;  S["Xte_r_sc"] = Xte_r_sc
    S["X_test_c"]  = X_test_c;  S["Xte_c_sc"] = Xte_c_sc
    S["yr_test"]   = yr_test;   S["yc_test"]  = yc_test
    S["scaler"]    = scaler;    S["X_cols"]   = ALL_FEATS

    def _already_reg(name): return any(r["Model"]==name for r in S["reg_results"])
    def _already_clf(name): return any(r["Model"]==name for r in S["clf_results"])

    def _train_reg(name, model):
        use_sc = name in ["Linear Regression","Ridge","Lasso"]
        Xtr = Xtr_r_sc if use_sc else X_train_r
        Xte = Xte_r_sc if use_sc else X_test_r
        t0  = time.time()
        model.fit(Xtr, yr_train)
        preds   = model.predict(Xte)
        elapsed = round(time.time()-t0, 2)
        row = {"Model":name,
               "R²":   round(r2_score(yr_test, preds),4),
               "MAE":  round(mean_absolute_error(yr_test, preds),2),
               "RMSE": round(np.sqrt(mean_squared_error(yr_test, preds)),2),
               "Time(s)": elapsed}
        # replace if already exists, else append
        S["reg_results"] = [r for r in S["reg_results"] if r["Model"]!=name] + [row]
        S["reg_models"][name] = model
        return row

    def _train_clf(name, model):
        use_sc = name in ["Logistic Regression","SVM (Linear)","KNN"]
        Xtr = Xtr_c_sc if use_sc else X_train_c
        Xte = Xte_c_sc if use_sc else X_test_c
        t0  = time.time()
        model.fit(Xtr, yc_train)
        preds   = model.predict(Xte)
        proba   = model.predict_proba(Xte)[:,1] if hasattr(model,"predict_proba") else None
        elapsed = round(time.time()-t0, 2)
        row = {"Model":name,
               "Accuracy":  round(accuracy_score(yc_test, preds),4),
               "F1":        round(f1_score(yc_test, preds, zero_division=0),4),
               "Precision": round(precision_score(yc_test, preds, zero_division=0),4),
               "Recall":    round(recall_score(yc_test, preds, zero_division=0),4),
               "ROC-AUC":   round(roc_auc_score(yc_test, proba),4) if proba is not None else 0.0,
               "Time(s)":   elapsed}
        S["clf_results"] = [r for r in S["clf_results"] if r["Model"]!=name] + [row]
        S["clf_models"][name] = model
        return row

    # ══════════════════════════════════════════════════════════
    # REGRESSION — individual buttons
    # ══════════════════════════════════════════════════════════
    st.markdown("---")
    sec("📈 Regression Models — Train Individually")
    info("Train each model one at a time. Results accumulate in Tab 2.")

    reg_cols = st.columns(3)
    reg_items = list(REG_MODELS.items())
    for idx, (name, model) in enumerate(reg_items):
        col = reg_cols[idx % 3]
        with col:
            trained = _already_reg(name)
            label   = f"✅ {name}" if trained else f"▶ Train {name}"
            if st.button(label, key=f"reg_{name}", use_container_width=True):
                with st.spinner(f"Training {name}..."):
                    row = _train_reg(name, model)
                st.success(f"R²={row['R²']:.4f} · MAE={row['MAE']:.2f} · {row['Time(s)']}s")
                st.rerun()
            if trained:
                r = next(r for r in S["reg_results"] if r["Model"]==name)
                st.caption(f"R²={r['R²']:.4f} · MAE={r['MAE']:.2f} · {r['Time(s)']}s")

    # live summary table
    if S["reg_results"]:
        st.markdown("---")
        reg_summary = pd.DataFrame(S["reg_results"]).sort_values("R²",ascending=False).reset_index(drop=True)
        S["reg_results_df"] = reg_summary
        st.dataframe(reg_summary.style.background_gradient(subset=["R²"],cmap="RdYlGn")
                                       .format({"R²":"{:.4f}","MAE":"{:.2f}","RMSE":"{:.2f}"}),
                     use_container_width=True)

    # ══════════════════════════════════════════════════════════
    # CLASSIFICATION — individual buttons
    # ══════════════════════════════════════════════════════════
    st.markdown("---")
    sec("🎯 Classification Models — Train Individually")
    info("Train each classifier one at a time. SVM (Linear) is fast — RBF SVM replaced to avoid freezing.")

    clf_cols = st.columns(3)
    clf_items = list(CLF_MODELS.items())
    for idx, (name, model) in enumerate(clf_items):
        col = clf_cols[idx % 3]
        with col:
            trained = _already_clf(name)
            label   = f"✅ {name}" if trained else f"▶ Train {name}"
            if st.button(label, key=f"clf_{name}", use_container_width=True):
                with st.spinner(f"Training {name}..."):
                    row = _train_clf(name, model)
                st.success(f"F1={row['F1']:.4f} · AUC={row['ROC-AUC']:.4f} · {row['Time(s)']}s")
                st.rerun()
            if trained:
                r = next(r for r in S["clf_results"] if r["Model"]==name)
                st.caption(f"F1={r['F1']:.4f} · AUC={r['ROC-AUC']:.4f} · {r['Time(s)']}s")

    # live summary table
    if S["clf_results"]:
        st.markdown("---")
        clf_summary = pd.DataFrame(S["clf_results"]).sort_values("F1",ascending=False).reset_index(drop=True)
        S["clf_results_df"] = clf_summary
        st.dataframe(clf_summary.style.background_gradient(subset=["F1","ROC-AUC"],cmap="RdYlGn")
                                        .format({c:"{:.4f}" for c in ["Accuracy","F1","Precision","Recall","ROC-AUC"]}),
                     use_container_width=True)

    # all done?
    n_trained = len(S["reg_results"]) + len(S["clf_results"])
    if n_trained == 12:
        st.success("✅ All 12 models trained! Navigate to Results tabs →")
    else:
        st.info(f"📊 {n_trained}/12 models trained so far.")

# ════════════════════════════════════════════════════════════
# TAB 2 — REGRESSION RESULTS
# ════════════════════════════════════════════════════════════
with tabs[1]:
    sec("📈 Tab 2 — Regression Results")
    info(f"Predicting: **{REG_TARGET}** · Metrics: R², MAE, RMSE")

    if not S.get("reg_results"):
        warn("Train at least one Regression model in Tab 1.")
    else:
        reg_df = pd.DataFrame(S["reg_results"]).sort_values("R²",ascending=False).reset_index(drop=True)

        # Metrics table
        st.dataframe(reg_df.style.background_gradient(subset=["R²"], cmap="RdYlGn")
                                  .background_gradient(subset=["MAE","RMSE"], cmap="RdYlGn_r")
                                  .format({"R²":"{:.4f}","MAE":"{:.2f}","RMSE":"{:.2f}"}),
                     use_container_width=True)

        best_reg = reg_df.iloc[0]["Model"]
        st.markdown(f"🏆 **Best Model:** `{best_reg}` — R² = {reg_df.iloc[0]['R²']:.4f}")

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            sec("📊 R² Comparison")
            fig = px.bar(reg_df, x="Model", y="R²",
                         color="R²", color_continuous_scale=["#c62828","#e65100","#2e7d32"],
                         title="R² Score — All Regression Models",
                         text=reg_df["R²"].apply(lambda x: f"{x:.4f}"))
            fig.update_traces(textposition="outside")
            fig.update_layout(height=380, xaxis_tickangle=-25)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            sec("📊 MAE & RMSE Comparison")
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(name="MAE",  x=reg_df["Model"], y=reg_df["MAE"],
                                  marker_color=CLR["warning"]))
            fig2.add_trace(go.Bar(name="RMSE", x=reg_df["Model"], y=reg_df["RMSE"],
                                  marker_color=CLR["danger"]))
            fig2.update_layout(barmode="group", height=380,
                                title="MAE vs RMSE — All Models",
                                xaxis_tickangle=-25)
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("---")
        sec("📈 Actual vs Predicted — Best Model")
        best_model = S["reg_models"][best_reg]
        use_sc     = best_reg in ["Linear Regression","Ridge","Lasso"]
        Xte        = S["Xte_r_sc"] if use_sc else S["X_test_r"]
        preds      = best_model.predict(Xte)
        yr_test    = S["yr_test"]

        col3, col4 = st.columns(2)
        with col3:
            fig3, ax = plt.subplots(figsize=(7,5))
            ax.scatter(yr_test, preds, alpha=0.3, s=8,
                       color=CLR["primary"], label="Predictions")
            lims = [min(yr_test.min(), preds.min()), max(yr_test.max(), preds.max())]
            ax.plot(lims, lims, "r--", lw=2, label="Perfect fit")
            ax.set_xlabel("Actual Amount ($)"); ax.set_ylabel("Predicted ($)")
            ax.set_title(f"Actual vs Predicted — {best_reg}")
            ax.legend(); plt.tight_layout(); st.pyplot(fig3); plt.close()

        with col4:
            residuals = yr_test.values - preds
            fig4, ax2 = plt.subplots(figsize=(7,5))
            ax2.scatter(preds, residuals, alpha=0.3, s=8, color=CLR["teal"])
            ax2.axhline(0, color=CLR["danger"], lw=2, ls="--")
            ax2.set_xlabel("Predicted ($)"); ax2.set_ylabel("Residual")
            ax2.set_title("Residual Plot")
            plt.tight_layout(); st.pyplot(fig4); plt.close()

        insight(f"Best regression model: {best_reg} with R²={reg_df.iloc[0]['R²']:.4f}.")
        warn("Regression on Transaction_Amount is secondary — main goal is fraud classification.")

# ════════════════════════════════════════════════════════════
# TAB 3 — CLASSIFICATION RESULTS
# ════════════════════════════════════════════════════════════
with tabs[2]:
    sec("🎯 Tab 3 — Classification Results")
    fraud_alert("Evaluating with F1, Precision, Recall, ROC-AUC — NOT accuracy. Class imbalance = 4.9%.")

    if not S.get("clf_results"):
        warn("Train at least one Classification model in Tab 1.")
    else:
        clf_df   = pd.DataFrame(S["clf_results"]).sort_values("F1",ascending=False).reset_index(drop=True)
        yc_test  = S["yc_test"]
        best_clf = clf_df.iloc[0]["Model"]

        st.dataframe(clf_df.style.background_gradient(subset=["F1","ROC-AUC"], cmap="RdYlGn")
                                  .background_gradient(subset=["Accuracy"], cmap="Blues")
                                  .format({c:"{:.4f}" for c in ["Accuracy","F1","Precision","Recall","ROC-AUC"]}),
                     use_container_width=True)

        st.markdown(f"🏆 **Best Classifier:** `{best_clf}` — F1={clf_df.iloc[0]['F1']:.4f} · AUC={clf_df.iloc[0]['ROC-AUC']:.4f}")

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            sec("📊 F1 Score Comparison")
            fig = px.bar(clf_df, x="Model", y="F1",
                         color="F1", color_continuous_scale=["#c62828","#e65100","#2e7d32"],
                         title="F1 Score — All Classifiers",
                         text=clf_df["F1"].apply(lambda x: f"{x:.4f}"))
            fig.update_traces(textposition="outside")
            fig.update_layout(height=380, xaxis_tickangle=-25)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            sec("📊 ROC-AUC Comparison")
            fig2 = px.bar(clf_df, x="Model", y="ROC-AUC",
                          color="ROC-AUC", color_continuous_scale=["#c62828","#e65100","#2e7d32"],
                          title="ROC-AUC — All Classifiers",
                          text=clf_df["ROC-AUC"].apply(lambda x: f"{x:.4f}"))
            fig2.update_traces(textposition="outside")
            fig2.update_layout(height=380, xaxis_tickangle=-25)
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("---")
        col3, col4 = st.columns(2)

        # Confusion Matrix — Best Model
        with col3:
            sec(f"🔢 Confusion Matrix — {best_clf}")
            best_m  = S["clf_models"][best_clf]
            use_sc  = best_clf in ["Logistic Regression","SVM (Linear)","KNN"]
            Xte_c   = S["Xte_c_sc"] if use_sc else S["X_test_c"]
            preds_c = best_m.predict(Xte_c)
            cm      = confusion_matrix(yc_test, preds_c)
            fig3, ax = plt.subplots(figsize=(5,4))
            import seaborn as sns
            sns.heatmap(cm, annot=True, fmt="d", cmap="Reds",
                        xticklabels=["Legit","Fraud"],
                        yticklabels=["Legit","Fraud"], ax=ax)
            ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
            ax.set_title(f"Confusion Matrix — {best_clf}")
            plt.tight_layout(); st.pyplot(fig3); plt.close()

            tn,fp,fn,tp = cm.ravel()
            st.markdown(f"**TP={tp}** (fraud caught) · **FN={fn}** (missed fraud) · "
                        f"**FP={fp}** (false alarm) · **TN={tn}** (correct legit)")

        # ROC Curve — Best Model
        with col4:
            sec(f"📈 ROC Curve — {best_clf}")
            if hasattr(best_m, "predict_proba"):
                proba_c = best_m.predict_proba(Xte_c)[:,1]
                fpr, tpr, _ = roc_curve(yc_test, proba_c)
                auc_val = roc_auc_score(yc_test, proba_c)
                fig4, ax2 = plt.subplots(figsize=(5,4))
                ax2.plot(fpr, tpr, color=CLR["danger"], lw=2.5,
                         label=f"ROC AUC = {auc_val:.4f}")
                ax2.plot([0,1],[0,1], color=CLR["grey"], ls="--", lw=1.5, label="Random")
                ax2.fill_between(fpr, tpr, alpha=0.1, color=CLR["danger"])
                ax2.set_xlabel("False Positive Rate")
                ax2.set_ylabel("True Positive Rate")
                ax2.set_title(f"ROC Curve — {best_clf}")
                ax2.legend(); plt.tight_layout(); st.pyplot(fig4); plt.close()

        st.markdown("---")
        sec("📊 Precision vs Recall — All Models")
        fig5 = go.Figure()
        fig5.add_trace(go.Scatter(
            x=clf_df["Recall"], y=clf_df["Precision"],
            mode="markers+text",
            text=clf_df["Model"],
            textposition="top center",
            marker=dict(size=clf_df["F1"]*30, color=clf_df["F1"],
                        colorscale="RdYlGn", showscale=True,
                        colorbar=dict(title="F1")),
        ))
        fig5.update_layout(title="Precision vs Recall (bubble size = F1)",
                           xaxis_title="Recall (Fraud Caught)",
                           yaxis_title="Precision (Alarm Quality)",
                           height=420)
        st.plotly_chart(fig5, use_container_width=True)

        insight(f"Best classifier: {best_clf} — highest F1 score for fraud detection.")
        fraud_alert(f"FN={fn} missed frauds — in real deployment, reducing FN is top priority over FP.")
        warn("High accuracy alone is misleading — a model predicting all 'Legit' gets 95.1% accuracy.")

# ════════════════════════════════════════════════════════════
# TAB 4 — FEATURE IMPORTANCE
# ════════════════════════════════════════════════════════════
with tabs[3]:
    sec("🔑 Tab 4 — Feature Importance")

    if not S.get("clf_models"):
        warn("Train at least one model in Tab 1.")
    else:
        clf_df  = pd.DataFrame(S["clf_results"]).sort_values("F1",ascending=False).reset_index(drop=True)
        reg_df  = pd.DataFrame(S["reg_results"]).sort_values("R²",ascending=False).reset_index(drop=True)
        feats   = S["X_cols"]

        col1, col2 = st.columns(2)

        # Classification importance
        with col1:
            sec("🎯 Classification — Feature Importance")
            best_clf = clf_df.iloc[0]["Model"]
            best_cm  = S["clf_models"][best_clf]
            if hasattr(best_cm, "feature_importances_"):
                imp = pd.DataFrame({"Feature": feats,
                                    "Importance": best_cm.feature_importances_})\
                        .sort_values("Importance", ascending=True)
                fig, ax = plt.subplots(figsize=(7, max(5, len(imp)*0.35)))
                colors_imp = [CLR["danger"] if i >= len(imp)-3 else CLR["primary"]
                              for i in range(len(imp))]
                ax.barh(imp["Feature"], imp["Importance"], color=colors_imp)
                ax.set_xlabel("Importance Score")
                ax.set_title(f"Feature Importance — {best_clf}")
                plt.tight_layout(); st.pyplot(fig); plt.close()
            elif hasattr(best_cm, "coef_"):
                coef = pd.DataFrame({"Feature": feats,
                                     "Coefficient": np.abs(best_cm.coef_[0])})\
                         .sort_values("Coefficient", ascending=True)
                fig, ax = plt.subplots(figsize=(7, max(5, len(coef)*0.35)))
                ax.barh(coef["Feature"], coef["Coefficient"], color=CLR["primary"])
                ax.set_xlabel("|Coefficient|")
                ax.set_title(f"Feature Coefficients — {best_clf}")
                plt.tight_layout(); st.pyplot(fig); plt.close()
            else:
                info(f"{best_clf} does not expose feature importances directly.")

        # Regression importance
        with col2:
            sec("📈 Regression — Feature Importance")
            best_reg = reg_df.iloc[0]["Model"]
            best_rm  = S["reg_models"][best_reg]
            if hasattr(best_rm, "feature_importances_"):
                imp2 = pd.DataFrame({"Feature": feats,
                                     "Importance": best_rm.feature_importances_})\
                         .sort_values("Importance", ascending=True)
                fig2, ax2 = plt.subplots(figsize=(7, max(5, len(imp2)*0.35)))
                colors_imp2 = [CLR["warning"] if i >= len(imp2)-3 else CLR["teal"]
                               for i in range(len(imp2))]
                ax2.barh(imp2["Feature"], imp2["Importance"], color=colors_imp2)
                ax2.set_xlabel("Importance Score")
                ax2.set_title(f"Feature Importance — {best_reg}")
                plt.tight_layout(); st.pyplot(fig2); plt.close()
            elif hasattr(best_rm, "coef_"):
                coef2 = pd.DataFrame({"Feature": feats,
                                      "Coefficient": np.abs(best_rm.coef_)})\
                          .sort_values("Coefficient", ascending=True)
                fig2, ax2 = plt.subplots(figsize=(7, max(5, len(coef2)*0.35)))
                ax2.barh(coef2["Feature"], coef2["Coefficient"], color=CLR["teal"])
                ax2.set_xlabel("|Coefficient|")
                ax2.set_title(f"Feature Coefficients — {best_reg}")
                plt.tight_layout(); st.pyplot(fig2); plt.close()
            else:
                info(f"{best_reg} does not expose feature importances directly.")

        st.markdown("---")
        sec("📊 Top Features — Combined View")
        if hasattr(S["clf_models"][clf_df.iloc[0]["Model"]], "feature_importances_"):
            clf_imp = pd.DataFrame({
                "Feature":    feats,
                "CLF_Imp":    S["clf_models"][clf_df.iloc[0]["Model"]].feature_importances_,
            })
            reg_m = S["reg_models"][reg_df.iloc[0]["Model"]]
            if hasattr(reg_m, "feature_importances_"):
                clf_imp["REG_Imp"] = reg_m.feature_importances_
            clf_imp = clf_imp.sort_values("CLF_Imp", ascending=False).head(10)
            fig3 = px.bar(clf_imp, x="Feature", y="CLF_Imp",
                          color="CLF_Imp",
                          color_continuous_scale=["#e3f2fd","#1565c0","#0d47a1"],
                          title="Top 10 Features for Fraud Classification",
                          labels={"CLF_Imp":"Importance Score"})
            fig3.update_layout(height=380, xaxis_tickangle=-20)
            st.plotly_chart(fig3, use_container_width=True)

        insight("Previous_Fraudulent_Transactions typically ranks #1 — past behavior is the strongest signal.")
        insight("Number_of_Transactions_Last_24H (velocity) usually ranks #2 — card testing pattern.")
        warn("KNN and SVM (Linear) don't expose feature importances — use RF/GB for interpretability.")

# ════════════════════════════════════════════════════════════
# TAB 5 — PREDICT
# ════════════════════════════════════════════════════════════
with tabs[4]:
    sec("🔮 Tab 5 — Interactive Fraud Risk Prediction")

    if not S.get("clf_models"):
        warn("Train at least one model in Tab 1.")
    else:
        clf_df = pd.DataFrame(S["clf_results"]).sort_values("F1",ascending=False).reset_index(drop=True)
        reg_df = pd.DataFrame(S["reg_results"]).sort_values("R²",ascending=False).reset_index(drop=True)
        info("Enter transaction details below to get a fraud risk score.")

        st.markdown("---")
        col1, col2, col3 = st.columns(3)

        with col1:
            sec("💳 Transaction Details")
            amount     = st.number_input("Transaction Amount ($)", 0.0, 50000.0, 500.0, 50.0)
            tx_type    = st.selectbox("Transaction Type",
                                       ["ATM Withdrawal","Bill Payment","POS Payment",
                                        "Bank Transfer","Online Purchase"])
            pay_method = st.selectbox("Payment Method",
                                       ["Debit Card","Credit Card","UPI",
                                        "Net Banking","Unknown"])
            device     = st.selectbox("Device Used",
                                       ["Mobile","Desktop","Tablet",
                                        "Unknown Device","Unknown"])
            location   = st.selectbox("Location",
                                       ["New York","San Francisco","Chicago",
                                        "Boston","Houston","Unknown"])

        with col2:
            sec("👤 User Profile")
            prev_fraud = st.number_input("Previous Fraudulent Transactions", 0, 20, 0)
            acct_age   = st.number_input("Account Age (days)", 0, 5000, 365)
            velocity   = st.number_input("Transactions in Last 24H", 0, 100, 3)
            hour       = st.slider("Hour of Transaction", 0, 23, 14)

        with col3:
            sec("📊 Computed Flags")
            is_night     = 1 if hour in [*range(22,24), *range(0,6)] else 0
            is_biz       = 1 if 9 <= hour <= 17 else 0
            q75_vel      = float(df["Number_of_Transactions_Last_24H"].quantile(0.75))
            q25_age      = float(df["Account_Age"].quantile(0.25))
            high_vel     = 1 if velocity > q75_vel else 0
            new_acc      = 1 if acct_age < q25_age else 0

            st.metric("Is Night",          "🌙 Yes" if is_night else "☀️ No")
            st.metric("Business Hours",    "✅ Yes" if is_biz   else "❌ No")
            st.metric("High Velocity",     "🚨 Yes" if high_vel else "✅ No")
            st.metric("New Account",       "⚠️ Yes" if new_acc  else "✅ No")
            st.metric("Velocity Q75",      f"{q75_vel:.0f} tx/24H")

        st.markdown("---")
        if st.button("🔮 Predict Fraud Risk", type="primary", use_container_width=True):
            # Encode categoricals using same mapping as training data
            le = LabelEncoder()
            def encode_cat(col_name, value):
                known = df[col_name].astype(str).unique().tolist()
                all_vals = known + [value] if value not in known else known
                le.fit(all_vals)
                return int(le.transform([value])[0])

            tx_enc  = encode_cat("Transaction_Type", tx_type)
            dev_enc = encode_cat("Device_Used",      device)
            loc_enc = encode_cat("Location",         location)
            pm_enc  = encode_cat("Payment_Method",   pay_method)

            input_row = pd.DataFrame([{
                "Previous_Fraudulent_Transactions": prev_fraud,
                "Account_Age":                      acct_age,
                "Number_of_Transactions_Last_24H":  velocity,
                "Hour":                             hour,
                "is_night":                         is_night,
                "is_business_hours":                is_biz,
                "high_velocity":                    high_vel,
                "new_account":                      new_acc,
                "Transaction_Type_enc":             tx_enc,
                "Device_Used_enc":                  dev_enc,
                "Location_enc":                     loc_enc,
                "Payment_Method_enc":               pm_enc,
            }])[S["X_cols"]]

            scaler_fit = S["scaler"]
            input_sc   = scaler_fit.transform(input_row)

            st.markdown("---")
            sec("🎯 Prediction Results — All Classifiers")
            pred_rows = []
            for name, model in S["clf_models"].items():
                use_sc = name in ["Logistic Regression","SVM (Linear)","KNN"]
                Xin    = input_sc if use_sc else input_row
                pred   = model.predict(Xin)[0]
                prob   = model.predict_proba(Xin)[0][1] if hasattr(model,"predict_proba") else None
                pred_rows.append({
                    "Model":       name,
                    "Prediction":  "🚨 FRAUD" if pred==1 else "✅ LEGIT",
                    "Fraud Prob%": f"{prob*100:.1f}%" if prob is not None else "N/A",
                    "Confidence":  f"{max(prob,1-prob)*100:.1f}%" if prob is not None else "N/A"
                })
            pred_df = pd.DataFrame(pred_rows)
            st.dataframe(pred_df, use_container_width=True)

            # Majority vote
            fraud_votes = sum(1 for r in pred_rows if "FRAUD" in r["Prediction"])
            total_votes = len(pred_rows)
            verdict     = "🚨 FRAUD DETECTED" if fraud_votes > total_votes/2 else "✅ LIKELY LEGITIMATE"
            color       = "error" if "FRAUD" in verdict else "success"

            st.markdown("---")
            sec("🏛 Final Verdict — Majority Vote")
            if "FRAUD" in verdict:
                fraud_alert(f"{verdict} — {fraud_votes}/{total_votes} models flagged this transaction.")
            else:
                insight(f"{verdict} — only {fraud_votes}/{total_votes} models flagged this transaction.")

            # Risk factors
            st.markdown("---")
            sec("⚠️ Risk Factors Identified")
            risks = []
            if prev_fraud > 0:  risks.append(f"Previous fraud history: {prev_fraud} incident(s)")
            if high_vel:        risks.append(f"High velocity: {velocity} tx/24H (threshold: {q75_vel:.0f})")
            if new_acc:         risks.append(f"New account: {acct_age} days old (threshold: {q25_age:.0f})")
            if is_night:        risks.append(f"Night-time transaction: {hour:02d}:00")
            if device in ["Unknown Device","Unknown"]: risks.append(f"Unknown device: {device}")
            if pay_method == "Unknown": risks.append("Unknown payment method")
            if amount > df["Transaction_Amount"].quantile(0.90):
                risks.append(f"High amount: ${amount:,.0f} (top 10%)")

            if risks:
                for r in risks:
                    warn(r)
            else:
                insight("No major risk factors detected for this transaction.")
