from pathlib import Path

import joblib
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
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "kNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Random Forest": "random_forest.joblib",
}

FEATURE_NAMES = [
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean",
    "smoothness_mean", "compactness_mean", "concavity_mean",
    "concave_points_mean", "symmetry_mean", "fractal_dimension_mean",
    "radius_se", "texture_se", "perimeter_se", "area_se",
    "smoothness_se", "compactness_se", "concavity_se",
    "concave_points_se", "symmetry_se", "fractal_dimension_se",
    "radius_worst", "texture_worst", "perimeter_worst", "area_worst",
    "smoothness_worst", "compactness_worst", "concavity_worst",
    "concave_points_worst", "symmetry_worst", "fractal_dimension_worst",
]

st.set_page_config(
    page_title="ML Classification Model Comparison",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 Machine Learning Classification Models")
st.subheader("Breast Cancer Wisconsin (Diagnostic) Dataset")

st.write(
    "Upload the test CSV, select a model, or select **All Models** "
    "to compare all five classifiers and identify the best-performing model."
)

uploaded_file = st.file_uploader(
    "Upload test data (CSV)",
    type=["csv"],
)

default_file = BASE / "test_data.csv"

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
elif default_file.exists():
    st.info(
        "Using the project's default test_data.csv. "
        "You can upload another test CSV above."
    )
    df = pd.read_csv(default_file)
else:
    st.warning("Please upload test_data.csv.")
    st.stop()

required_columns = FEATURE_NAMES + ["target"]
missing = [c for c in required_columns if c not in df.columns]

if missing:
    st.error(
        "The uploaded CSV is missing required columns: "
        + ", ".join(missing)
    )
    st.stop()

X_test = df[FEATURE_NAMES]
y_test = df["target"].astype(int)

model_options = ["All Models"] + list(MODEL_FILES.keys())

selected_model = st.selectbox(
    "Select Machine Learning Model",
    model_options,
)


def evaluate_model(model_name):
    model_path = MODEL_DIR / MODEL_FILES[model_name]

    if not model_path.exists():
        return None

    model = joblib.load(model_path)
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, pred),
        "AUC": roc_auc_score(y_test, proba),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1 Score": f1_score(y_test, pred, zero_division=0),
        "MCC": matthews_corrcoef(y_test, pred),
    }

    return model, pred, metrics


if selected_model == "All Models":

    st.markdown("## 📊 Comparison of All Models")

    all_results = []
    model_outputs = {}

    for model_name in MODEL_FILES:
        result = evaluate_model(model_name)

        if result is None:
            st.error(f"Trained model not found for {model_name}.")
            continue

        model, pred, metrics = result
        all_results.append(metrics)
        model_outputs[model_name] = {"model": model, "pred": pred}

    if not all_results:
        st.stop()

    results_df = pd.DataFrame(all_results)

    metric_columns = [
        "Accuracy", "AUC", "Precision",
        "Recall", "F1 Score", "MCC"
    ]

    # Best method = highest mean of the six required metrics.
    results_df["Mean Score"] = results_df[metric_columns].mean(axis=1)

    results_df = results_df.sort_values(
        "Mean Score",
        ascending=False
    ).reset_index(drop=True)

    results_df["Rank"] = range(1, len(results_df) + 1)

    best_model = results_df.iloc[0]["Model"]
    best_score = results_df.iloc[0]["Mean Score"]

    st.success(
        f"🏆 **Best Method: {best_model}**  \n"
        f"Mean Score: **{best_score:.4f}**"
    )

    st.markdown("### Model Performance Comparison")

    display_df = results_df[
        [
            "Rank", "Model", "Accuracy", "AUC",
            "Precision", "Recall", "F1 Score",
            "MCC", "Mean Score"
        ]
    ].copy()

    st.dataframe(
        display_df.style.format({
            "Accuracy": "{:.4f}",
            "AUC": "{:.4f}",
            "Precision": "{:.4f}",
            "Recall": "{:.4f}",
            "F1 Score": "{:.4f}",
            "MCC": "{:.4f}",
            "Mean Score": "{:.4f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### 🏆 Best Model Metrics")

    best_row = results_df.iloc[0]

    c1, c2, c3 = st.columns(3)
    c4, c5, c6 = st.columns(3)

    c1.metric("Accuracy", f"{best_row['Accuracy']:.4f}")
    c2.metric("AUC", f"{best_row['AUC']:.4f}")
    c3.metric("Precision", f"{best_row['Precision']:.4f}")
    c4.metric("Recall", f"{best_row['Recall']:.4f}")
    c5.metric("F1 Score", f"{best_row['F1 Score']:.4f}")
    c6.metric("MCC", f"{best_row['MCC']:.4f}")

    st.markdown("## 🔎 Detailed Results for All Models")

    for model_name in results_df["Model"]:
        st.markdown(f"### {model_name}")

        pred = model_outputs[model_name]["pred"]
        cm = confusion_matrix(y_test, pred)

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Confusion Matrix**")

            cm_df = pd.DataFrame(
                cm,
                index=["Actual Benign", "Actual Malignant"],
                columns=["Predicted Benign", "Predicted Malignant"],
            )

            st.dataframe(cm_df, use_container_width=True)

        with col2:
            st.write("**Classification Report**")

            report = classification_report(
                y_test,
                pred,
                target_names=["Benign", "Malignant"],
                output_dict=True,
                zero_division=0,
            )

            st.dataframe(
                pd.DataFrame(report).transpose(),
                use_container_width=True,
            )

else:

    result = evaluate_model(selected_model)

    if result is None:
        st.error(
            f"Trained model not found for {selected_model}. "
            "Run train_models.py first."
        )
        st.stop()

    model, pred, metrics = result

    st.markdown(
        f"## 📈 {selected_model} - Evaluation Results"
    )

    c1, c2, c3 = st.columns(3)
    c4, c5, c6 = st.columns(3)

    c1.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
    c2.metric("AUC", f"{metrics['AUC']:.4f}")
    c3.metric("Precision", f"{metrics['Precision']:.4f}")
    c4.metric("Recall", f"{metrics['Recall']:.4f}")
    c5.metric("F1 Score", f"{metrics['F1 Score']:.4f}")
    c6.metric("MCC", f"{metrics['MCC']:.4f}")

    st.markdown("### Confusion Matrix")

    cm = confusion_matrix(y_test, pred)

    cm_df = pd.DataFrame(
        cm,
        index=["Actual Benign", "Actual Malignant"],
        columns=["Predicted Benign", "Predicted Malignant"],
    )

    st.dataframe(cm_df, use_container_width=True)

    st.markdown("### Classification Report")

    report = classification_report(
        y_test,
        pred,
        target_names=["Benign", "Malignant"],
        output_dict=True,
        zero_division=0,
    )

    st.dataframe(
        pd.DataFrame(report).transpose(),
        use_container_width=True,
    )

st.markdown("## 📋 Test Data Preview")
st.dataframe(df.head(10), use_container_width=True)
