from preprocessing.prepare_data import process_data
import os
from pathlib import Path
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def _resolve_path(p: str | None) -> str | None:
    if not p:
        return None
    return p if os.path.isabs(p) else os.path.join(PROJECT_ROOT, p)

# Load data
data = process_data(
    train_transaction_path = _resolve_path("data/train_transaction.csv"),
    train_identity_path = _resolve_path("data/train_identity.csv")
)

X_train, X_test, y_train, y_test = data

# Work on copies
X_train = X_train.copy()
X_test = X_test.copy()

# Find categorical columns
cat_cols = X_train.select_dtypes(include=["object", "category", "str"]).columns

# Frequency encoding based only on training data
for col in cat_cols:
    freq_map = X_train[col].value_counts(normalize=True, dropna=False)
    X_train[col] = X_train[col].map(freq_map)
    X_test[col] = X_test[col].map(freq_map)

# Fill missing values and unseen test categories
X_train = X_train.fillna(0)
X_test = X_test.fillna(0)

# Make sure train and test have identical columns
X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)

print("Shapes after encoding:")
print("X_train:", X_train.shape)
print("X_test :", X_test.shape)

# Scale features for MLP
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Fast MLP baseline for large data
mlp = MLPClassifier(
    hidden_layer_sizes=(64, 32),   
    activation="relu",
    solver="adam",                 
    batch_size=512,                
    learning_rate_init=0.001,
    max_iter=20,                   
    early_stopping=True,           
    validation_fraction=0.1,
    n_iter_no_change=3,
    random_state=42,
    verbose=True
)

mlp.fit(X_train, y_train)

y_pred = mlp.predict(X_test)
y_proba = mlp.predict_proba(X_test)[:, 1]

print("\n--- MLP Classifier ---")
print("Precision:", precision_score(y_test, y_pred, zero_division=0))
print("Recall   :", recall_score(y_test, y_pred, zero_division=0))
print("F1-score :", f1_score(y_test, y_pred, zero_division=0))
print("ROC-AUC  :", roc_auc_score(y_test, y_proba))