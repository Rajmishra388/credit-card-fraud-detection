"""
Baseline ML models trained on the RAW imbalanced data (no resampling).
Purpose: establish baseline performance before we address class imbalance
with SMOTE/class-weighting in later scripts (bagging/boosting stage).

Models: Logistic Regression, Decision Tree, K-Nearest Neighbors
"""

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from utils import load_processed_data, evaluate_and_log

RANDOM_STATE = 42

# ---- Load data ----
print("Loading processed data...")
X_train, X_test, y_train, y_test = load_processed_data()

# ---- Model 1: Logistic Regression ----
print("\nTraining Logistic Regression...")
lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
y_proba = lr.predict_proba(X_test)[:, 1]
evaluate_and_log("logistic_regression", y_test, y_pred, y_proba)

# ---- Model 2: Decision Tree ----
print("\nTraining Decision Tree...")
dt = DecisionTreeClassifier(random_state=RANDOM_STATE)
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)
y_proba = dt.predict_proba(X_test)[:, 1]
evaluate_and_log("decision_tree", y_test, y_pred, y_proba)

# ---- Model 3: K-Nearest Neighbors ----
print("\nTraining K-Nearest Neighbors...")
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
y_proba = knn.predict_proba(X_test)[:, 1]
evaluate_and_log("k_nearest_neighbors", y_test, y_pred, y_proba)

print("\nBaseline training complete. Check results/metrics.csv for the comparison table.")