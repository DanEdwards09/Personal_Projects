"""
Compact business-rules module: compute a normalized business risk score in [0,1]
for each applicant row (pandas Series). Also helper to apply to a DataFrame.
"""

import numpy as np

# Simple categorical mappings (0 = low risk, 1 = high risk). Tune as needed.
_CHECKING_MAP = {
    'A11': 1.0,  # < 0 DM
    'A12': 0.7,  # 0 <= ... < 200
    'A13': 0.2,  # >= 200 / salary assignments
    'A14': 0.8,  # no checking account
}
_CREDIT_HISTORY_MAP = {
    'A30': 0.1, 'A31': 0.1, 'A32': 0.2, 'A33': 0.6, 'A34': 1.0
}
_SAVINGS_MAP = {
    'A61': 0.9, 'A62': 0.6, 'A63': 0.4, 'A64': 0.2, 'A65': 0.6
}
_EMPLOYMENT_MAP = {
    'A71': 1.0, 'A72': 0.8, 'A73': 0.4, 'A74': 0.2, 'A75': 0.1
}

def _map_cat(mapping, key):
    return mapping.get(key, 0.5)

def compute_business_score(row, max_credit_amount):
    """
    Compute a weighted business risk score (0 low risk -> 1 high risk) for a single row.
    row: pandas Series containing named columns used in ML_model.py (checking_account_status, credit_history, ...)
    max_credit_amount: numeric used to normalize credit amount into [0,1]
    """
    # Categorical parts
    s_check = _map_cat(_CHECKING_MAP, row.get('checking_account_status'))
    s_hist = _map_cat(_CREDIT_HISTORY_MAP, row.get('credit_history'))
    s_sav = _map_cat(_SAVINGS_MAP, row.get('savings_account'))
    s_emp = _map_cat(_EMPLOYMENT_MAP, row.get('employment_since'))

    # Numerical parts (normalized to [0,1])
    credit_amount = float(row.get('credit_amount', 0) or 0)
    s_credit_amt = min(credit_amount / float(max_credit_amount or 1), 1.0)

    age = float(row.get('age', 0) or 0)
    if age < 25:
        s_age = 0.6
    elif age < 40:
        s_age = 0.3
    elif age < 60:
        s_age = 0.2
    else:
        s_age = 0.5

    num_credits = float(row.get('num_existing_credits', 0) or 0)
    s_existing = min(num_credits / 3.0, 1.0)  # >3 credits => high risk

    # Weights (sum to 1)
    weights = {
        'checking': 0.20,
        'history': 0.25,
        'savings': 0.15,
        'employment': 0.10,
        'credit_amount': 0.15,
        'existing_credits': 0.10,
        'age': 0.05
    }

    biz_score = (
        weights['checking'] * s_check +
        weights['history'] * s_hist +
        weights['savings'] * s_sav +
        weights['employment'] * s_emp +
        weights['credit_amount'] * s_credit_amt +
        weights['existing_credits'] * s_existing +
        weights['age'] * s_age
    )

    # Ensure in [0,1]
    return float(np.clip(biz_score, 0.0, 1.0))


def compute_business_scores(df):
    """
    Apply compute_business_score across a DataFrame and return a numpy array of scores.
    """
    max_credit = df['credit_amount'].max() if 'credit_amount' in df.columns else 1.0
    return df.apply(lambda r: compute_business_score(r, max_credit), axis=1).to_numpy()
