import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score


# 1. Load the dataset from a shared Google Drive link

df = pd.read_csv('https://drive.google.com/file/d/1Q5zoXlmJoM1hVXELtWZF1g8VSTzoLg1a/view?usp=sharing')
# Display the first few rows of the DataFrame
print(df.head())


# 2. Feature Engineering

# Create 'hour' from the 'Time' column (assuming Time is in seconds)
df['hour'] = ((df['Time'] // 3600) % 24).astype(int)

# Create 'transactions per hour' by counting the number of transactions in the same hour
df['tx_per_hour'] = df.groupby('hour')['Time'].transform('count')

# Drop the original 'Time' column as it's been transformed
df.drop('Time', axis=1, inplace=True)


# 3. Prepare Features and Target

# Assume PCA-transformed variables are V1-V28 and include 'Amount', 'hour', and 'tx_per_hour'
feature_cols = ['Amount'] + [f'V{i}' for i in range(1, 29)] + ['hour', 'tx_per_hour']
X = df[feature_cols]
y = df['Class']

# Split the data (using stratification due to class imbalance)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# 4. Logistic Regression with GridSearchCV

param_grid = {
    'C': [0.01, 0.1, 1],
    'penalty': ['l1', 'l2'],
    'class_weight': [None, 'balanced']
}

# Using the liblinear solver which supports both L1 and L2 penalties
lr = LogisticRegression(solver='liblinear', max_iter=1000, random_state=42)

grid_search = GridSearchCV(lr, param_grid, scoring='roc_auc', cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

best_lr = grid_search.best_estimator_
print("Best Logistic Regression Parameters:", grid_search.best_params_)


# 5. Model Evaluation

y_pred = best_lr.predict(X_test)
y_proba = best_lr.predict_proba(X_test)[:, 1]

print("Logistic Regression Evaluation Metrics:")
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("AUC-ROC:", roc_auc_score(y_test, y_proba))
print("F1 Score:", f1_score(y_test, y_pred))