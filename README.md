# 🔍 P7 — Fraud Detection Analysis & Prediction
**M3 · ML Engine Portfolio · Project 7 of 12**

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit)](https://streamlit.io)
[![Dataset](https://img.shields.io/badge/Source-Kaggle-20BEFF?logo=kaggle)](https://www.kaggle.com)

---

## 📌 Project Overview

End-to-end fraud detection pipeline on **51,000 financial transactions** — tackling one of the most critical challenges in fintech: identifying fraudulent transactions in a severely imbalanced dataset (4.9% fraud).

**Core Questions:**
- Which transaction behaviors best predict fraud?
- Does high velocity (rapid transactions) significantly differ from normal? (A/B Test)
- Can we detect fraud outliers statistically? (Anomaly Detection)
- Can ML models reliably catch fraud while minimizing false alarms?

---

## ⚠️ Critical Challenge — Class Imbalance

| Class | Count | Rate |
|-------|-------|------|
| Legitimate | 48,490 | 95.1% |
| Fraudulent | 2,510 | 4.9% |

A naive model predicting **"all legitimate"** achieves **95% accuracy but catches 0 fraud**.

**Solution:** `class_weight='balanced'` applied to all classifiers. Evaluate with **F1, Precision, Recall, ROC-AUC**.

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| Source | Kaggle Fraud Detection Dataset |
| Records | 51,000 transactions |
| Original Features | 12 |
| After Engineering | 18+ |
| Nulls | ~5% in 5 columns — imputed |

---

## 🗂 Project Structure

```
📁 Repo_7_Fraud_Detection/
├── Home.py
├── M3_logo.png
├── requirements.txt
├── README.md
├── data/
│   └── fraud_clean.csv        ← from P7_clean_data.py
└── pages/
    ├── EDA_dashboard.py       ← 13-tab analysis
    └── ML_Models.py           ← 5-tab ML engine
```

---

## 📈 EDA Dashboard — 13 Tabs

| # | Tab | Highlight |
|---|-----|-----------|
| 1 | Data Overview | Dictionary, stats, class balance |
| 2 | Fraud Distribution ★ | Fraud rate by type + payment method |
| 3 | Amount Analysis ★ | Distribution + Pareto 80/20 |
| 4 | Transaction Patterns ★ | Velocity + account age behavioral analysis |
| 5 | Time Analysis ★ | Hourly fraud rate + night vs day |
| 6 | Device & Location ★ | Fraud rate by device and city |
| 7 | Feature Engineering | Engineered flags + distributions |
| 8 | Correlation | Full heatmap + top fraud predictors |
| 9 | Anomaly Detection ★ | Z-score scatter + IQR outlier table |
| 10 | A/B Test ★ | High velocity vs normal — Welch T-test + Cohen's d |
| 11 | Missing Values | Imputation strategy per column |
| 12 | Multicollinearity | VIF analysis |
| 13 | Insights & Report | Findings + recommendations + download |

---

## 🤖 ML Models — 5 Tabs

| Tab | Content |
|-----|---------|
| 1 | Training — 6 Regression + 6 Classification models |
| 2 | Regression Results — R², MAE, RMSE · predict Transaction_Amount |
| 3 | Classification Results — F1, Precision, Recall, ROC-AUC · confusion matrix · ROC curve |
| 4 | Feature Importance — top fraud predictors |
| 5 | Interactive Predict — real-time fraud risk scoring with majority vote |

### Models Used
**Regression (6):** Linear · Ridge · Lasso · Decision Tree · Random Forest · Gradient Boosting

**Classification (6):** Logistic Regression · Decision Tree · Random Forest · Gradient Boosting · SVM · KNN

---

## 🔑 Key Findings

**1. Previous Fraud History is #1 Signal**
Users with past fraud incidents show dramatically higher current fraud rates. Past behavior predicts future risk.

**2. Transaction Velocity = Card Testing**
High velocity (transactions > 75th percentile in 24H) is a classic card testing attack pattern.

**3. New Accounts = Higher Risk**
Accounts below the 25th percentile in age show disproportionately higher fraud — fraudsters open fresh accounts.

**4. Night-Time Fraud Elevation**
Transactions between 10 PM – 6 AM show elevated fraud rates when human oversight is minimal.

**5. Unknown Device = Red Flag**
Unrecognized devices trigger high fraud rates — device fingerprinting is a strong control.

---

## 💡 Recommendations

| Priority | Action |
|----------|--------|
| 🚨 Critical | Real-time ML scoring on every transaction |
| ⚡ High | Flag velocity > 75th percentile for review |
| 🆕 High | Transaction caps on new accounts (Age < Q25) |
| 🌙 Medium | Automated scrutiny 10 PM – 6 AM |
| 📱 Medium | Unknown device = OTP / biometric step-up |
| 💳 Medium | Flag Unknown / Invalid payment methods |
| 📜 High | Weight previous fraud history in rule engine |

---

## 🚀 How to Run

```bash
git clone https://github.com/YourUsername/Repo_7_Fraud_Detection.git
cd Repo_7_Fraud_Detection
pip install -r requirements.txt

# Step 1: Generate clean dataset
# Run P7_clean_data.py in Jupyter → produces fraud_clean.csv
# Place fraud_clean.csv in data/ folder

# Step 2: Launch app
streamlit run Home.py
```

> Upload `fraud_clean.csv` via the sidebar uploader, or place it in `data/` for auto-load.

---

## 🛠 Tech Stack

`Python 3.11` · `Streamlit` · `Pandas` · `NumPy` · `Matplotlib` · `Seaborn` · `Plotly` · `Scikit-learn` · `SciPy` · `Statsmodels` · `Psutil`

---

**Mohamed · M3 · ML Engine Portfolio — 12 End-to-End Data Science Projects**
