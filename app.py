from pathlib import Path
import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score

BASE = Path(__file__).resolve().parent
MODEL_DIR = BASE / 'model'
MODEL_FILES = {
    'Logistic Regression': 'logistic_regression.joblib',
    'Decision Tree': 'decision_tree.joblib',
    'kNN': 'knn.joblib',
    'Naive Bayes': 'naive_bayes.joblib',
    'Random Forest': 'random_forest.joblib',
}
FEATURE_NAMES = [
    'radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean','compactness_mean','concavity_mean','concave_points_mean','symmetry_mean','fractal_dimension_mean',
    'radius_se','texture_se','perimeter_se','area_se','smoothness_se','compactness_se','concavity_se','concave_points_se','symmetry_se','fractal_dimension_se',
    'radius_worst','texture_worst','perimeter_worst','area_worst','smoothness_worst','compactness_worst','concavity_worst','concave_points_worst','symmetry_worst','fractal_dimension_worst'
]

st.set_page_config(page_title='ML Classification Model Comparison', page_icon='🧠', layout='wide')
st.title('🧠 Machine Learning Classification Models')
st.subheader('Breast Cancer Wisconsin (Diagnostic) Dataset')
st.write('Upload test data, select a trained model, and view the required evaluation metrics.')

uploaded_file = st.file_uploader('Upload test data (CSV)', type=['csv'])
default_file = BASE / 'test_data.csv'
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
elif default_file.exists():
    st.info('Using the project test_data.csv. You can upload another test CSV above.')
    df = pd.read_csv(default_file)
else:
    st.warning('Please upload test_data.csv.')
    st.stop()

required = FEATURE_NAMES + ['target']
missing = [c for c in required if c not in df.columns]
if missing:
    st.error('Missing required columns: ' + ', '.join(missing))
    st.stop()

X_test = df[FEATURE_NAMES]
y_test = df['target'].astype(int)
selected_model = st.selectbox('Select Machine Learning Model', list(MODEL_FILES.keys()))
model_path = MODEL_DIR / MODEL_FILES[selected_model]
if not model_path.exists():
    st.error(f'Trained model not found: {model_path}. Run train_models.py first.')
    st.stop()

model = joblib.load(model_path)
pred = model.predict(X_test)
proba = model.predict_proba(X_test)[:, 1]
accuracy = accuracy_score(y_test, pred)
auc = roc_auc_score(y_test, proba)
precision = precision_score(y_test, pred, zero_division=0)
recall = recall_score(y_test, pred, zero_division=0)
f1 = f1_score(y_test, pred, zero_division=0)
mcc = matthews_corrcoef(y_test, pred)

st.markdown('### Evaluation Metrics')
c1,c2,c3 = st.columns(3)
c4,c5,c6 = st.columns(3)
c1.metric('Accuracy', f'{accuracy:.4f}')
c2.metric('AUC', f'{auc:.4f}')
c3.metric('Precision', f'{precision:.4f}')
c4.metric('Recall', f'{recall:.4f}')
c5.metric('F1 Score', f'{f1:.4f}')
c6.metric('MCC', f'{mcc:.4f}')

st.markdown('### Confusion Matrix')
cm = confusion_matrix(y_test, pred)
st.dataframe(pd.DataFrame(cm, index=['Actual Benign','Actual Malignant'], columns=['Predicted Benign','Predicted Malignant']), use_container_width=True)
st.markdown('### Classification Report')
report = classification_report(y_test, pred, target_names=['Benign','Malignant'], output_dict=True, zero_division=0)
st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)
st.markdown('### Test Data Preview')
st.dataframe(df.head(10), use_container_width=True)
