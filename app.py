import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

BASE = Path(__file__).resolve().parent
MODEL_DIR = BASE / "model"

MODEL_FILES = {
    "Logistic Regression": MODEL_DIR / "logistic_regression.joblib",
    "Decision Tree": MODEL_DIR / "decision_tree.joblib",
    "kNN": MODEL_DIR / "knn.joblib",
    "Naive Bayes": MODEL_DIR / "naive_bayes.joblib",
    "Random Forest": MODEL_DIR / "random_forest.joblib",
}

with open(MODEL_DIR / "metadata.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)

st.set_page_config(page_title="ML Classification Model Comparison", page_icon="🤖", layout="wide")

st.title("🤖 Machine Learning Classification Model Comparison")
st.write(
    "Upload the test CSV generated for Assignment–2, select a classification model, "
    "and view its evaluation metrics and confusion matrix."
)

st.sidebar.header("Model Selection")
selected_model = st.sidebar.selectbox("Choose a model", list(MODEL_FILES.keys()))

uploaded_file = st.file_uploader("Upload test_data.csv", type=["csv"])

if uploaded_file is None:
    st.info("Please upload the test_data.csv file to evaluate the selected model.")
    st.subheader("Dataset used")
    st.write("Breast Cancer Wisconsin (Diagnostic) — UCI Machine Learning Repository")
    st.write("569 instances, 30 numerical features, binary target")
    st.write("Target used in this project: 1 = malignant, 0 = benign")
    st.stop()

try:
    test_df = pd.read_csv(uploaded_file)
except Exception as exc:
    st.error(f"Could not read the CSV file: {exc}")
    st.stop()

required_columns = metadata["feature_names"] + ["target"]
missing = [c for c in required_columns if c not in test_df.columns]

if missing:
    st.error("The uploaded file is missing required columns.")
    st.write(missing)
    st.stop()

X_test = test_df[metadata["feature_names"]]
y_test = test_df["target"].astype(int)

model = joblib.load(MODEL_FILES[selected_model])
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

metrics = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "AUC": roc_auc_score(y_test, y_proba),
    "Precision": precision_score(y_test, y_pred, zero_division=0),
    "Recall": recall_score(y_test, y_pred, zero_division=0),
    "F1 Score": f1_score(y_test, y_pred, zero_division=0),
    "MCC": matthews_corrcoef(y_test, y_pred),
}

st.subheader(f"Results — {selected_model}")
cols = st.columns(6)
for col, (name, value) in zip(cols, metrics.items()):
    col.metric(name, f"{value:.4f}")

left, right = st.columns(2)

with left:
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    fig, ax = plt.subplots()
    ax.imshow(cm)
    ax.set_xticks([0, 1], labels=["Benign (0)", "Malignant (1)"])
    ax.set_yticks([0, 1], labels=["Benign (0)", "Malignant (1)"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(selected_model)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with right:
    st.subheader("Classification Report")
    report = classification_report(
        y_test,
        y_pred,
        target_names=["Benign", "Malignant"],
        output_dict=True,
        zero_division=0,
    )
    st.dataframe(pd.DataFrame(report).T.round(4), use_container_width=True)

st.subheader("All Model Comparison")
comparison = pd.DataFrame(metadata["results"]).set_index("Model")
st.dataframe(comparison.style.format("{:.4f}"), use_container_width=True)
st.success(f"Overall winner by mean of the six required metrics: {metadata['overall_winner']}")

st.caption("For academic assignment use only. This application demonstrates machine-learning classification workflow and is not a medical diagnostic tool.")
