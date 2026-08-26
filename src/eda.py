"""
Exploratory Data Analysis for Credit Card Fraud Detection dataset.
Generates and saves key plots to results/eda/ for use in the research paper.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ---- Config ----
DATA_PATH = "data/raw/creditcard.csv"
OUTPUT_DIR = "results/eda"

os.makedirs(OUTPUT_DIR, exist_ok=True)
sns.set_style("whitegrid")

# ---- Load data ----
print("Loading dataset...")
df = pd.read_csv(DATA_PATH)
print(f"Shape: {df.shape}")
print(df.head())

# ---- Basic info ----
print("\nMissing values per column:")
print(df.isnull().sum().sum(), "total missing values")

print("\nClass distribution:")
print(df["Class"].value_counts())
print(df["Class"].value_counts(normalize=True) * 100)

# ---- 1. Class imbalance plot ----
plt.figure(figsize=(6, 4))
sns.countplot(x="Class", data=df)
plt.title("Class Distribution (0 = Legit, 1 = Fraud)")
plt.xlabel("Class")
plt.ylabel("Count")
plt.yscale("log")  # log scale since fraud is so rare
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/class_distribution.png", dpi=150)
plt.close()
print(f"\nSaved: {OUTPUT_DIR}/class_distribution.png")

# ---- 2. Correlation heatmap ----
plt.figure(figsize=(16, 12))
corr = df.corr()
sns.heatmap(corr, cmap="coolwarm", center=0, annot=False)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/correlation_heatmap.png", dpi=150)
plt.close()
print(f"Saved: {OUTPUT_DIR}/correlation_heatmap.png")

# ---- 3. Amount distribution: Fraud vs Legit ----
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.histplot(df[df["Class"] == 0]["Amount"], bins=50, ax=axes[0], color="steelblue")
axes[0].set_title("Transaction Amount - Legit")
axes[0].set_xlim(0, 2000)

sns.histplot(df[df["Class"] == 1]["Amount"], bins=50, ax=axes[1], color="crimson")
axes[1].set_title("Transaction Amount - Fraud")
axes[1].set_xlim(0, 2000)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/amount_distribution.png", dpi=150)
plt.close()
print(f"Saved: {OUTPUT_DIR}/amount_distribution.png")

# ---- 4. Time distribution: Fraud vs Legit ----
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.histplot(df[df["Class"] == 0]["Time"], bins=50, ax=axes[0], color="steelblue")
axes[0].set_title("Transaction Time - Legit")

sns.histplot(df[df["Class"] == 1]["Time"], bins=50, ax=axes[1], color="crimson")
axes[1].set_title("Transaction Time - Fraud")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/time_distribution.png", dpi=150)
plt.close()
print(f"Saved: {OUTPUT_DIR}/time_distribution.png")

print("\nEDA complete. Check results/eda/ for saved plots.")