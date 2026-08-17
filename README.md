# Machine Learning Assignment – 2

## a. Problem Statement

The objective of this project is to implement and compare multiple supervised classification models on a common classification dataset. The models are evaluated using Accuracy, AUC, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC). An interactive Streamlit application is provided to upload test data, select a model, and display its evaluation results.

The assignment brief says “all 6 ML models” but lists five models and provides five rows in the comparison table. This implementation follows the five models explicitly listed in the brief: Logistic Regression, Decision Tree, kNN, Naive Bayes, and Random Forest.

## b. Dataset Description

**Dataset:** Breast Cancer Wisconsin (Diagnostic)

**Source:** UCI Machine Learning Repository

**Dataset characteristics:**
- 569 instances
- 30 numerical predictive features
- Binary classification
- Features are computed from digitized images of fine needle aspirates of breast masses.

The dataset satisfies the assignment minimum of 12 features and 500 instances.

The original UCI/scikit-learn target encoding is 0 = malignant and 1 = benign. For this project, the target is remapped to make the medically important class the positive class:

- `1 = malignant`
- `0 = benign`

Source: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic

## c. GitHub Repository Link

**GitHub Repository:** `https://github.com/anuradhadevieee/ml_assignment_2`

## Streamlit App Link

**Live Streamlit App:** `https://mlassignment2-wncyqpdzz4tsch4xynj5pj.streamlit.app/`

## d. Models Used

### Comparison of Evaluation Metrics

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9649 | 0.9960 | 0.9750 | 0.9286 | 0.9512 | 0.9245 |
| Decision Tree | 0.9211 | 0.9448 | 0.9459 | 0.8333 | 0.8861 | 0.8299 |
| kNN | 0.9561 | 0.9823 | 0.9744 | 0.9048 | 0.9383 | 0.9058 |
| Naive Bayes | 0.9386 | 0.9934 | 1.0000 | 0.8333 | 0.9091 | 0.8715 |
| Random Forest (Ensemble) | **0.9737** | 0.9944 | **1.0000** | 0.9286 | **0.9630** | **0.9442** |

### Observations on Model Performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Performs very well across all metrics. Its AUC is extremely high, indicating strong separation between the two classes. Standardization is used before classification. |
| Decision Tree | Gives the lowest overall performance among the five models. It still achieves good classification accuracy, but its recall and MCC are lower than the other models. |
| kNN | Provides strong performance with high accuracy, AUC, precision, recall, F1, and MCC. Feature scaling is important for distance-based classification. |
| Naive Bayes | Produces perfect precision on this test split and a very high AUC, but its recall is lower, meaning some malignant cases are not detected. |
| Random Forest (Ensemble) | Achieves the highest accuracy, precision, F1, and MCC among the tested models. Its balanced overall performance makes it the strongest model for this dataset and test split. |

### Overall Winner for the Dataset

**Random Forest** is the overall winner based on the highest mean score across the six required evaluation metrics. Its mean score is approximately **0.9673**, followed by Logistic Regression at approximately **0.9567**.

## Project Structure

```text
ML_Assignment_2_Project/
├── app.py
├── train_models.py
├── requirements.txt
├── README.md
├── test_data.csv
├── model_results.csv
├── .gitignore
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    ├── random_forest.joblib
    └── metadata.json
```

## How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal and upload `test_data.csv`.

## How the Models Were Trained

The project uses an 80:20 stratified train-test split with `random_state=42`. The training portion contains 455 instances and the test portion contains 114 instances.

Logistic Regression and kNN use StandardScaler. Decision Tree uses a maximum depth of 5. Random Forest uses 300 trees. The saved pipelines/models are stored in the `model/` directory.

## Reproducing the Models

Run:

```bash
python train_models.py
```

This recreates `test_data.csv`, the saved model files, `model_results.csv`, and `model/metadata.json`.

## Streamlit Features

The deployed application includes:

1. CSV upload for test data.
2. Model-selection dropdown.
3. Accuracy, AUC, Precision, Recall, F1, and MCC display.
4. Confusion matrix.
5. Classification report.
6. Comparison table for all five models.

## Academic Note

The application is intended for demonstrating a machine-learning classification workflow as part of the academic assignment. It is not a medical diagnostic system.
