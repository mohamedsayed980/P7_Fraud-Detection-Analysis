"""
P7 Fraud Detection — Data Cleaning & Feature Engineering
Run this in Jupyter ONCE → saves fraud_clean.csv
Mohamed · M3
"""
import pandas as pd
import numpy as np

# ── 1. LOAD ───────────────────────────────────────────────────
df = pd.read_csv("Fraud Detection Dataset.csv")
print(f"✅ Raw shape     : {df.shape}")
print(f"✅ Total nulls   : {df.isnull().sum().sum()}")

# ── 2. DROP ID COLUMNS ────────────────────────────────────────
df.drop(columns=["Transaction_ID", "User_ID"], inplace=True)
print(f"✅ After drop IDs: {df.shape}")

# ── 3. NUMERIC IMPUTATION ─────────────────────────────────────
df["Transaction_Amount"].fillna(df["Transaction_Amount"].median(), inplace=True)

# ── 4. TIME FEATURE ───────────────────────────────────────────
df.rename(columns={"Time_of_Transaction": "Hour"}, inplace=True)
df["Hour"].fillna(df["Hour"].median(), inplace=True)
df["Hour"] = df["Hour"].astype(int)

# ── 5. CATEGORICAL IMPUTATION ────────────────────────────────
for col in ["Device_Used", "Location"]:
    df[col].fillna("Unknown", inplace=True)

# Payment_Method: NaN + "Invalid Method" → "Unknown"
df["Payment_Method"].fillna("Unknown", inplace=True)
df["Payment_Method"] = df["Payment_Method"].replace("Invalid Method", "Unknown")

# ── 6. ENGINEERED FEATURES ───────────────────────────────────
df["is_night"]          = df["Hour"].isin([*range(22, 24), *range(0, 6)]).astype(int)
df["is_business_hours"] = df["Hour"].between(9, 17).astype(int)

df["high_velocity"] = (df["Number_of_Transactions_Last_24H"] >
                        df["Number_of_Transactions_Last_24H"].quantile(0.75)).astype(int)
df["new_account"]   = (df["Account_Age"] <
                        df["Account_Age"].quantile(0.25)).astype(int)

df["Amount_Category"] = pd.cut(
    df["Transaction_Amount"],
    bins=[0, 100, 500, 1000, 5000, 99999],
    labels=["Very Low", "Low", "Medium", "High", "Very High"]
)

# ── 7. LABEL ENCODING (for ML) ───────────────────────────────
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for col in ["Transaction_Type", "Device_Used", "Location", "Payment_Method"]:
    df[col + "_enc"] = le.fit_transform(df[col].astype(str))

# ── 8. FINAL CHECK ───────────────────────────────────────────
print(f"\n✅ Final shape   : {df.shape}")
print(f"✅ Total nulls   : {df.isnull().sum().sum()}")
print(f"✅ Columns       : {df.columns.tolist()}")
print(f"\n✅ Target balance:\n{df['Fraudulent'].value_counts()}")
print(f"\n✅ Fraud rate    : {df['Fraudulent'].mean()*100:.2f}%")
print(f"\n✅ Payment_Method values: {df['Payment_Method'].unique()}")

# ── 9. SAVE ──────────────────────────────────────────────────
df.to_csv("fraud_clean.csv", sep=",", decimal=".", index=False, encoding="utf-8")
print("\n✅ Saved: fraud_clean.csv — ready for Streamlit app")
