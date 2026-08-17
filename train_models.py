import json
from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

BASE = Path(__file__).resolve().parent
DATA_FILE = BASE / 'dataset' / 'wdbc.data'
MODEL_DIR = BASE / 'model'
MODEL_DIR.mkdir(exist_ok=True)

if not DATA_FILE.exists():
    raise FileNotFoundError(f'Dataset file not found: {DATA_FILE}')

feature_names = [
    'radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean','compactness_mean','concavity_mean','concave_points_mean','symmetry_mean','fractal_dimension_mean',
    'radius_se','texture_se','perimeter_se','area_se','smoothness_se','compactness_se','concavity_se','concave_points_se','symmetry_se','fractal_dimension_se',
    'radius_worst','texture_worst','perimeter_worst','area_worst','smoothness_worst','compactness_worst','concavity_worst','concave_points_worst','symmetry_worst','fractal_dimension_worst'
]
columns = ['id', 'diagnosis'] + feature_names

data = pd.read_csv(DATA_FILE, header=None, names=columns)
X = data[feature_names].copy()
y = (data['diagnosis'] == 'M').astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

test_df = X_test.copy()
test_df['target'] = y_test.values
test_df.to_csv(BASE / 'test_data.csv', index=False)

models = {
    'Logistic Regression': Pipeline([('scaler', StandardScaler()), ('classifier', LogisticRegression(max_iter=5000, random_state=42))]),
    'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
    'kNN': Pipeline([('scaler', StandardScaler()), ('classifier', KNeighborsClassifier(n_neighbors=5))]),
    'Naive Bayes': GaussianNB(),
    'Random Forest': RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
}

results = []
metadata = {
    'dataset': 'Breast Cancer Wisconsin (Diagnostic)',
    'source': 'UCI Machine Learning Repository',
    'dataset_file': 'dataset/wdbc.data',
    'instances': int(X.shape[0]),
    'features': int(X.shape[1]),
    'target_definition': {'1': 'Malignant', '0': 'Benign'},
    'feature_names': feature_names,
    'train_rows': int(len(X_train)),
    'test_rows': int(len(X_test)),
    'random_state': 42,
    'test_size': 0.20,
    'models': list(models.keys()),
    'evaluation_metrics': ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC'],
}

for name, model in models.items():
    print(f'Training: {name}')
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        'Model': name,
        'Accuracy': accuracy_score(y_test, pred),
        'AUC': roc_auc_score(y_test, proba),
        'Precision': precision_score(y_test, pred, zero_division=0),
        'Recall': recall_score(y_test, pred, zero_division=0),
        'F1': f1_score(y_test, pred, zero_division=0),
        'MCC': matthews_corrcoef(y_test, pred),
    }
    results.append(metrics)
    print(metrics)
    print('Confusion Matrix:')
    print(confusion_matrix(y_test, pred))
    filename = name.lower().replace(' ', '_') + '.joblib'
    joblib.dump(model, MODEL_DIR / filename)

results_df = pd.DataFrame(results)
metric_cols = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']
results_df['Mean Score'] = results_df[metric_cols].mean(axis=1)
winner = results_df.sort_values('Mean Score', ascending=False).iloc[0]['Model']
results_df.drop(columns=['Mean Score']).to_csv(BASE / 'model_results.csv', index=False)
metadata['overall_winner'] = winner
metadata['results'] = results_df.drop(columns=['Mean Score']).to_dict(orient='records')
with open(MODEL_DIR / 'metadata.json', 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=2)

print('\nDataset shape:', data.shape)
print('Train/Test:', X_train.shape, X_test.shape)
print('\nFinal model comparison:')
print(results_df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
print('\nOverall winner by mean of six metrics:', winner)
