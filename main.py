import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

# Variables
num_of_cv_splits = 5
# Load data
data = pd.read_csv('./dataset/LUSCexpfile.csv', sep=';', index_col=0, low_memory=False)
labels = data.iloc[0]
gene_expr = data.iloc[1:]

X = gene_expr.T.astype(float) # Get transpose of dataset because it is originally in form: cols = patients rows = genes
y = LabelEncoder().fit_transform(labels.values) # Encode labels to numerical values

# Set up models to compare
# Use smote
# For pipeline we use these attributes for preprocessing:
### scaler: standardizes gene expression values
### pca: decreases number of features to those with most variance (original data has over 56,000 genes)
### smote: "synthetic minority oversampling technique" -> creates synthetic normal samples using interpolation to balance dataset
### clf: classifier (model)
# NOTE: Using pipeline (from imbalance-learn package) ensures the preprocessing only occurs during training not testing
### This is because we need to generate normal samples using SMOTE but do not want those included in the test folds during cross-validation

models = {
    'Logistic Regression': Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=100)),
        ('smote', SMOTE(random_state=42)),
        ('clf', LogisticRegression(max_iter=1000, random_state=42)),
    ]),
    'Random Forest': Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=100)),
        ('smote', SMOTE(random_state=42)),
        ('clf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ]),
    'SVM': Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=100)),
        ('smote', SMOTE(random_state=42)),
        ('clf', SVC(kernel='rbf', probability=True, random_state=42)),
    ]),
    'XGBoost': Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=100)),
        ('smote', SMOTE(random_state=42)),
        ('clf', XGBClassifier(random_state=42, eval_metric='logloss', verbosity=0)),
    ]),
    'KNN': Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=100)),
        ('smote', SMOTE(random_state=42)),
        ('clf', KNeighborsClassifier(n_neighbors=5)),
    ]),
}

# Cross-validation
cross_val = StratifiedKFold(n_splits=num_of_cv_splits, shuffle=True, random_state=42)
# Model evaluation metrics
metrics = ['accuracy', 'f1', 'roc_auc']

results = {}
for name, model in models.items():
    print(f"Running model: {name}...")
    scores = cross_validate(model, X, y, cv=cross_val, scoring=metrics)
    results[name] = {metric: scores[f'test_{metric}'].mean() for metric in metrics}

# Print results
results_data = pd.DataFrame(results).T # Transpose results for charting
print('\n', results_data.sort_values('roc_auc', ascending=False).round(3))

# Generate bar graph of comparisons
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(results_data))
width = 0.20

for i, metric in enumerate(metrics):
    ax.bar(x + i * width, results_data[metric], width, label=metric)

ax.set_xticks(x + width)
ax.set_xticklabels(results_data.index, rotation=15, ha='right')
ax.set_ylabel('Score')
ax.set_title(f'Model Comparison ({num_of_cv_splits}-Fold Cross-Validation)')
ax.set_ylim(0.95, 1.01)
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150)
plt.show()