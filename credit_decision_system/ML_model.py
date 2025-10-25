from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report, accuracy_score
import joblib
from business_rules import compute_business_scores
  
# Fetch dataset from UCI repo
statlog_german_credit_data = fetch_ucirepo(id=144) 
  
# Split into features and target
X = statlog_german_credit_data.data.features 
y = statlog_german_credit_data.data.targets 
  
# Metadata 
print(statlog_german_credit_data.metadata) 
  
# Variable information 
print(statlog_german_credit_data.variables) 

# Rename columns for better understanding
X = X.rename(columns={
    # Categorical features
    'Attribute1': 'checking_account_status',
    'Attribute3': 'credit_history',
    'Attribute4': 'purpose',
    'Attribute6': 'savings_account',
    'Attribute7': 'employment_since',
    'Attribute9': 'personal_status_sex',
    'Attribute10': 'other_debtors',
    'Attribute12': 'property',
    'Attribute14': 'other_installment_plans',
    'Attribute15': 'housing',
    'Attribute17': 'job',
    'Attribute19': 'telephone',
    'Attribute20': 'foreign_worker',
    
    # Numerical features
    'Attribute2': 'duration_months',
    'Attribute5': 'credit_amount',
    'Attribute8': 'installment_rate',
    'Attribute11': 'residence_since',
    'Attribute13': 'age',
    'Attribute16': 'num_existing_credits',
    'Attribute18': 'num_dependents'
})

X.head()

# Prepare target y as 1/0
y = y.values.ravel()
y = y - 1  # original labels (1=good,2=bad) -> (0=good,1=bad)

# Identify feature types
categorical_features = X.select_dtypes(include=['object']).columns.tolist()
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Preprocessors
preprocessor_scaled = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features),
    ]
)

preprocessor_passthrough = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_features),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features),
    ]
)

pipelines = {
    'log_reg': Pipeline([('prep', preprocessor_scaled), ('model', LogisticRegression(max_iter=1000))]),
    'rf': Pipeline([('prep', preprocessor_passthrough), ('model', RandomForestClassifier(random_state=42))])
}

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train and evaluate models, then combine with business rules
results = {}
for name, pipe in pipelines.items():
    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    auc = roc_auc_score(y_test, proba)
    print(f"{name} AUC: {auc:.3f}")
    print(classification_report(y_test, pred, zero_division=0))
    results[name] = {'pipeline': pipe, 'proba': proba}

# Compute business-rule scores for X_test (aligned order)
biz_scores = compute_business_scores(X_test.reset_index(drop=True))

# Combine model probabilities with business score and evaluate
alpha = 0.7  # weight to give to model probability vs business rules (tuneable)

for name, info in results.items():
    model_proba = info['proba']
    # ensure same ordering alignment between model_proba and biz_scores
    if len(model_proba) != len(biz_scores):
        biz_scores = compute_business_scores(X_test.reset_index(drop=True))
    combined_score = alpha * model_proba + (1 - alpha) * biz_scores
    combined_pred = (combined_score >= 0.5).astype(int)

    print(f"--- {name} + business rules (alpha={alpha}) ---")
    print("Model-only AUC:", f"{roc_auc_score(y_test, model_proba):.3f}")
    print("Combined AUC (using combined score as ranking):", f"{roc_auc_score(y_test, combined_score):.3f}")
    print("Model-only Classification Report:")
    print(classification_report(y_test, (model_proba >= 0.5).astype(int), zero_division=0))
    print("Combined Classification Report:")
    print(classification_report(y_test, combined_pred, zero_division=0))

    # Optionally persist pipeline for later use
    joblib.dump(info['pipeline'], f"{name}_pipeline.joblib")