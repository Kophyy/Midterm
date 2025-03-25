import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score


# 1. Load the dataset from a shared Google Drive link
df = pd.read_csv('https://drive.google.com/file/d/1Q5zoXlmJoM1hVXELtWZF1g8VSTzoLg1a/view?usp=sharing')


# 2. Feature Engineering

# Create 'hour' from the 'Time' column (assuming Time is in seconds)
df['hour'] = ((df['Time'] // 3600) % 24).astype(int)

# Create 'transactions per hour' by counting transactions within the same hour
df['tx_per_hour'] = df.groupby('hour')['Time'].transform('count')

# Drop the original 'Time' column as it's no longer needed
df.drop('Time', axis=1, inplace=True)


# 3. Prepare Features and Target

# Features include Amount, PCA-transformed variables (V1-V28), hour, and tx_per_hour
feature_cols = ['Amount'] + [f'V{i}' for i in range(1, 29)] + ['hour', 'tx_per_hour']
X = df[feature_cols]
y = df['Class']

# Split the dataset into train and test sets using stratification due to class imbalance
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# 4. Random Forest with GridSearchCV

param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, 20],
    'min_samples_split': [2, 5],
    'class_weight': ['balanced', 'balanced_subsample']
}

rf = RandomForestClassifier(random_state=42)
grid_rf = GridSearchCV(rf, param_grid=param_grid_rf, scoring='roc_auc', cv=5, n_jobs=-1)
grid_rf.fit(X_train, y_train)

best_rf = grid_rf.best_estimator_
print("Best Random Forest Parameters:", grid_rf.best_params_)


# 5. Model Evaluation

y_pred_rf = best_rf.predict(X_test)
y_proba_rf = best_rf.predict_proba(X_test)[:, 1]

print("Random Forest Evaluation Metrics:")
print("Precision:", precision_score(y_test, y_pred_rf))
print("Recall:", recall_score(y_test, y_pred_rf))
print("AUC-ROC:", roc_auc_score(y_test, y_proba_rf))
print("F1 Score:", f1_score(y_test, y_pred_rf))