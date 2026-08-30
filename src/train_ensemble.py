"""
Ensemble models: Bagging (Random Forest, Bagging Classifier) and
Boosting (AdaBoost, Gradient Boosting, XGBoost).

Unlike the baseline script, this one addresses class imbalance using SMOTE
applied ONLY to the training set (never the test set, to avoid data leakage).
"""

from sklearn.ensemble import (
    RandomForestClassifier, BaggingClassifier,
    AdaBoostClassifier, GradientBoostingClassifier
)
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

from utils import load_processed_data, evaluate_and_log

RANDOM_STATE = 42

# ---- Load data ----
print("Loading processed data...")
X_train, X_test, y_train, y_test = load_processed_data()

# ---- Apply SMOTE to training data only ----
print("Applying SMOTE to training set...")
print(f"Before SMOTE: {y_train.value_counts().to_dict()}")
smote = SMOTE(random_state=RANDOM_STATE)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print(f"After SMOTE:  {y_train_res.value_counts().to_dict()}")

# ================= BAGGING =================

# ---- Model 1: Random Forest ----
print("\nTraining Random Forest...")
rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
rf.fit(X_train_res, y_train_res)
y_pred = rf.predict(X_test)
y_proba = rf.predict_proba(X_test)[:, 1]
evaluate_and_log("random_forest", y_test, y_pred, y_proba)

# ---- Model 2: Bagging Classifier (Decision Tree base) ----
print("\nTraining Bagging Classifier...")
bag = BaggingClassifier(
    estimator=DecisionTreeClassifier(random_state=RANDOM_STATE),
    n_estimators=50, random_state=RANDOM_STATE, n_jobs=-1
)
bag.fit(X_train_res, y_train_res)
y_pred = bag.predict(X_test)
y_proba = bag.predict_proba(X_test)[:, 1]
evaluate_and_log("bagging_classifier", y_test, y_pred, y_proba)

# ================= BOOSTING =================

# ---- Model 3: AdaBoost ----
print("\nTraining AdaBoost...")
ada = AdaBoostClassifier(random_state=RANDOM_STATE)
ada.fit(X_train_res, y_train_res)
y_pred = ada.predict(X_test)
y_proba = ada.predict_proba(X_test)[:, 1]
evaluate_and_log("adaboost", y_test, y_pred, y_proba)

# ---- Model 4: Gradient Boosting ----
print("\nTraining Gradient Boosting...")
gb = GradientBoostingClassifier(random_state=RANDOM_STATE)
gb.fit(X_train_res, y_train_res)
y_pred = gb.predict(X_test)
y_proba = gb.predict_proba(X_test)[:, 1]
evaluate_and_log("gradient_boosting", y_test, y_pred, y_proba)

# ---- Model 5: XGBoost ----
print("\nTraining XGBoost...")
xgb = XGBClassifier(
    eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1
)
xgb.fit(X_train_res, y_train_res)
y_pred = xgb.predict(X_test)
y_proba = xgb.predict_proba(X_test)[:, 1]
evaluate_and_log("xgboost", y_test, y_pred, y_proba)

print("\nEnsemble training complete. Check results/metrics.csv for the full comparison table.")
