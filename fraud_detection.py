# Import libraries  
import pandas as pd  
from sklearn.model_selection import train_test_split, GridSearchCV  
from sklearn.ensemble import RandomForestClassifier  
from sklearn.linear_model import LogisticRegression  
from sklearn.metrics import precision_score, recall_score, roc_auc_score

# Constants 
DATA_URL = "https://drive.google.com/file/d/1Q5zoXlmJoM1hVXELtWZF1g8VSTzoLg1a/view?usp=sharing"  
RANDOM_SEED = 42

# Functions  
def load_data(url: str) -> pd.DataFrame:
    """Load data from URL."""
    df = pd.read_csv('https://drive.google.com/file/d/1Q5zoXlmJoM1hVXELtWZF1g8VSTzoLg1a/view?usp=sharing') 
    return df
# Display the first few rows of the DataFrame
    print(df.head())

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:  
    """Add time-based features."""  
    df['hour'] = (df['Time'] // 3600) % 24  # Convert seconds to hour  
    df['txn_per_hour'] = df.groupby('hour')['hour'].transform('count')  
    return df

def train_model(X_train, y_train, model, params):  
    """Hyperparameter tuning with GridSearchCV."""  
    grid = GridSearchCV(model, params, scoring='roc_auc', cv=3)  
    grid.fit(X_train, y_train)  
    return grid.best_estimator_

# Load data  
df = load_data('https://drive.google.com/file/d/1Q5zoXlmJoM1hVXELtWZF1g8VSTzoLg1a/view?usp=sharing')  
df = engineer_features(df)

# Split data  
X = df.drop('Class', axis=1)  
y = df['Class']  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)  

# Train models  
logreg_params = {'C': [0.01, 0.1, 1], 'penalty': ['l1', 'l2'], 'class_weight': [None, 'balanced']}  
rf_params = {'n_estimators': [100, 200], 'max_depth': [5, 10, 20], 'min_samples_split': [2, 5]}  

best_logreg = train_model(X_train, y_train, LogisticRegression(), logreg_params)  
best_rf = train_model(X_train, y_train, RandomForestClassifier(), rf_params)  

# Evaluate  
for model in [best_logreg, best_rf]:  
    y_pred = model.predict(X_test)  
    print(f"Precision: {precision_score(y_test, y_pred):.2f}")  
    print(f"Recall: {recall_score(y_test, y_pred):.2f}")  