import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

BASE = Path(__file__).resolve().parent
MODEL_DIR = BASE / "model"
MODEL_DIR.mkdir(exist_ok=True)

# UCI Breast Cancer Wisconsin (Diagnostic), available through sklearn as a copy.
data = load_breast_cancer(as_frame=True)
X = data.data.copy()
# Original sklearn/UCI target: 0=malignant, 1=benign.
# For this assignment, make malignant the positive class: 1=malignant, 0=benign.
y = (data.target == 0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

test_df = X_test.copy()
test_df["target"] = y_test.values
test_df.to_csv(BASE / "test_data.csv", index=False)

models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=5000, random_state=42)),
    ]),
    "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=5),
    "kNN": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", KNeighborsClassifier(n_neighbors=5)),
    ]),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, random_state=42, n_jobs=-1
    ),
}

results = []
metadata = {
    "dataset": "Breast Cancer Wisconsin (Diagnostic)",
    "source": "UCI Machine Learning Repository",
    "target_definition": "1 = malignant, 0 = benign",
    "feature_names": list(X.columns),
    "test_rows": len(X_test),
    "train_rows": len(X_train),
    "random_state": 42,
    "test_size": 0.20,
}

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "AUC": roc_auc_score(y_test, proba),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1": f1_score(y_test, pred, zero_division=0),
        "MCC": matthews_corrcoef(y_test, pred),
    }
    results.append(metrics)

    filename = name.lower().replace(" ", "_") + ".joblib"
    joblib.dump(model, MODEL_DIR / filename)

results_df = pd.DataFrame(results)
results_df.to_csv(BASE / "model_results.csv", index=False)

# Rank by mean of the six required metrics.
metric_cols = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
results_df["Mean Score"] = results_df[metric_cols].mean(axis=1)
winner = results_df.sort_values("Mean Score", ascending=False).iloc[0]["Model"]
metadata["overall_winner"] = winner
metadata["results"] = results_df.drop(columns=["Mean Score"]).to_dict(orient="records")
with open(MODEL_DIR / "metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

print("Dataset shape:", X.shape)
print("Train/Test:", X_train.shape, X_test.shape)
print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print("Overall winner by mean of six metrics:", winner)
