"""
Preprocessing for Credit Card Fraud Detection dataset.
- Scales Time and Amount
- Performs stratified train/test split
- Saves processed splits to data/processed/ for use by all model scripts
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

# ---- Config ----
DATA_PATH = "data/raw/creditcard.csv"
OUTPUT_DIR = "data/processed"
TEST_SIZE = 0.2
RANDOM_STATE = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---- Load data ----
print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

# ---- Scale Time and Amount ----
print("Scaling 'Time' and 'Amount'...")
scaler = StandardScaler()
df["Time"] = scaler.fit_transform(df[["Time"]])
df["Amount"] = scaler.fit_transform(df[["Amount"]])

# ---- Split features and target ----
X = df.drop(columns=["Class"])
y = df["Class"]

# ---- Stratified train/test split ----
print(f"Splitting data (test_size={TEST_SIZE}, stratified on Class)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    stratify=y,
    random_state=RANDOM_STATE
)

print(f"Train shape: {X_train.shape}, Fraud in train: {y_train.sum()} ({y_train.mean()*100:.3f}%)")
print(f"Test shape:  {X_test.shape}, Fraud in test:  {y_test.sum()} ({y_test.mean()*100:.3f}%)")

# ---- Save processed data ----
X_train.to_csv(f"{OUTPUT_DIR}/X_train.csv", index=False)
X_test.to_csv(f"{OUTPUT_DIR}/X_test.csv", index=False)
y_train.to_csv(f"{OUTPUT_DIR}/y_train.csv", index=False)
y_test.to_csv(f"{OUTPUT_DIR}/y_test.csv", index=False)

# ---- Save the scaler too (so it can be reused for new/unseen data later if needed) ----
joblib.dump(scaler, f"{OUTPUT_DIR}/scaler.pkl")

print(f"\nPreprocessing complete. Files saved to {OUTPUT_DIR}/")