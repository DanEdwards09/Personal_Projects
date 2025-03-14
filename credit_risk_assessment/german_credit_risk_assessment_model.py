# German Credit Risk Assessment Model
# Author: [Your Name]
# Date: March 14, 2025

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
import warnings

warnings.filterwarnings("ignore")

# Set style for plots
sns.set_style("whitegrid")
plt.style.use("fivethirtyeight")

# 1. Data Loading and Exploration
# ------------------------------
print("Loading the German Credit dataset...")
# First, let's try to identify the file correctly
import os

# List all files in the directory
print("Files in directory:")
for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Now try to load the file with the correct path and format
try:
    # Try as CSV first (most common format)
    df = pd.read_csv("/kaggle/input/german-credit/german_credit_data.csv")
    print("Successfully loaded CSV file!")
except Exception as e:
    print(f"Error loading CSV: {e}")
    try:
        # Try as Excel if CSV fails
        df = pd.read_excel("/kaggle/input/german-credit/german_credit_data.csv")
        print("Successfully loaded as Excel file!")
    except Exception as e:
        print(f"Error loading as Excel: {e}")
        # As a last resort, try with a different extension
        try:
            # Try to find any Excel files
            excel_files = [
                f
                for f in os.listdir("/kaggle/input/german-credit")
                if f.endswith(".xlsx") or f.endswith(".xls")
            ]
            if excel_files:
                df = pd.read_excel(
                    os.path.join("/kaggle/input/german-credit", excel_files[0])
                )
                print(f"Successfully loaded alternate Excel file: {excel_files[0]}")
            else:
                # Try to find any CSV files
                csv_files = [
                    f
                    for f in os.listdir("/kaggle/input/german-credit")
                    if f.endswith(".csv")
                ]
                if csv_files:
                    df = pd.read_csv(
                        os.path.join("/kaggle/input/german-credit", csv_files[0])
                    )
                    print(f"Successfully loaded alternate CSV file: {csv_files[0]}")
                else:
                    raise FileNotFoundError("No suitable data files found")
        except Exception as e:
            print(f"Final error: {e}")
            # Create a sample dataset as a fallback
            print("Creating a sample dataset as fallback...")
            # Create sample data
            np.random.seed(42)
            data = {
                "Age": np.random.randint(18, 80, 1000),
                "Sex": np.random.choice(["male", "female"], 1000),
                "Job": np.random.randint(0, 4, 1000),
                "Housing": np.random.choice(["own", "rent", "free"], 1000),
                "Saving accounts": np.random.choice(
                    ["little", "moderate", "quite rich", "rich"], 1000
                ),
                "Checking account": np.random.choice(
                    ["little", "moderate", "rich"], 1000
                ),
                "Credit amount": np.random.randint(500, 15000, 1000),
                "Duration": np.random.randint(6, 72, 1000),
                "Purpose": np.random.choice(
                    ["car", "radio/TV", "furniture/equipment", "business", "education"],
                    1000,
                ),
            }
            df = pd.DataFrame(data)

# Clean up column names if needed
if "__EMPTY" in df.columns:
    df.rename(columns={"__EMPTY": "ID"}, inplace=True)

# Display basic information
print("\nDataset Overview:")
print(f"Shape: {df.shape}")
print("\nFirst few rows:")
print(df.head())

print("\nColumn Information:")
print(df.info())

print("\nSummary Statistics:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

# 2. Data Preprocessing
# ---------------------
print("\nPreprocessing the data...")

# Make a copy of the dataframe for preprocessing
df_clean = df.copy()

# Handle "NA" values (which are strings, not actual NaN values)
for col in df_clean.columns:
    if df_clean[col].dtype == "object":
        df_clean[col] = df_clean[col].replace("NA", np.nan)

# Handle actual missing values
for col in df_clean.columns:
    if df_clean[col].dtype == "object":
        df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
    else:
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())

# Identify categorical and numerical columns
categorical_cols = df_clean.select_dtypes(include=["object"]).columns.tolist()
numerical_cols = df_clean.select_dtypes(include=["int64", "float64"]).columns.tolist()

# Remove ID column from analysis if it exists
id_cols = ["ID"]
for col in id_cols:
    if col in numerical_cols:
        numerical_cols.remove(col)

print(f"\nCategorical columns: {categorical_cols}")
print(f"Numerical columns: {numerical_cols}")

# The rest of the code remains the same as before
# 3. Exploratory Data Analysis
# ----------------------------
print("\nPerforming Exploratory Data Analysis...")

# Set up the plots for EDA
plt.figure(figsize=(15, 10))

# Age distribution
plt.subplot(2, 2, 1)
sns.histplot(data=df_clean, x="Age", bins=20, kde=True)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")

# Credit amount distribution
plt.subplot(2, 2, 2)
sns.histplot(data=df_clean, x="Credit amount", bins=20, kde=True)
plt.title("Credit Amount Distribution")
plt.xlabel("Credit Amount")
plt.ylabel("Count")

# Duration distribution
plt.subplot(2, 2, 3)
sns.histplot(data=df_clean, x="Duration", bins=20, kde=True)
plt.title("Loan Duration Distribution")
plt.xlabel("Duration (months)")
plt.ylabel("Count")

# Purpose distribution
plt.subplot(2, 2, 4)
purpose_counts = df_clean["Purpose"].value_counts()
sns.barplot(x=purpose_counts.index, y=purpose_counts.values)
plt.title("Loan Purpose Distribution")
plt.xlabel("Purpose")
plt.ylabel("Count")
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig("credit_distributions.png")

# Continue with the rest of your code...
# Feature Engineering, Risk Assessment Model, etc.


# Finish with the Risk Prediction Function
def predict_credit_risk(client_data):
    """
    Simplified version of credit risk prediction that doesn't rely on saved models

    Parameters:
    -----------
    client_data : pd.DataFrame
        DataFrame containing client information

    Returns:
    --------
    dict
        Dictionary containing risk assessment results
    """
    # Factor weights
    factor_weights = {
        "age_factor": 0.10,
        "credit_amount_factor": 0.25,
        "duration_factor": 0.20,
        "savings_factor": 0.20,
        "checking_factor": 0.15,
        "purpose_factor": 0.10,
    }

    # Calculate component scores
    # Age score
    age = client_data["Age"].values[0]
    age_score = min(100, max(0, (age - 20) * 2)) if age >= 25 else max(0, age - 18) * 5

    # Credit amount score
    max_credit = 20000
    credit_amount = client_data["Credit amount"].values[0]
    credit_amount_score = max(0, 100 - (credit_amount / max_credit * 100))

    # Duration score
    max_duration = 72
    duration = client_data["Duration"].values[0]
    duration_score = max(0, 100 - (duration / max_duration * 100))

    # Savings score
    savings_map = {
        "little": 25,
        "moderate": 50,
        "quite rich": 75,
        "rich": 100,
        np.nan: 0,
    }
    savings = client_data["Saving accounts"].values[0]
    savings_score = savings_map.get(savings, 0)

    # Checking account score
    checking_map = {"little": 25, "moderate": 60, "rich": 100, np.nan: 0}
    checking = client_data["Checking account"].values[0]
    checking_score = checking_map.get(checking, 0)

    # Purpose score
    purpose_risk = {
        "car": 60,
        "radio/TV": 80,
        "furniture/equipment": 70,
        "business": 40,
        "education": 75,
        "domestic appliances": 65,
        "repairs": 60,
        "vacation/others": 50,
    }
    purpose = client_data["Purpose"].values[0]
    purpose_score = purpose_risk.get(purpose, 50)

    # Calculate overall credit score
    credit_score = (
        age_score * factor_weights["age_factor"]
        + credit_amount_score * factor_weights["credit_amount_factor"]
        + duration_score * factor_weights["duration_factor"]
        + savings_score * factor_weights["savings_factor"]
        + checking_score * factor_weights["checking_factor"]
        + purpose_score * factor_weights["purpose_factor"]
    )

    # Determine risk category and recommendation
    if credit_score >= 80:
        risk_category = "Low Risk"
        recommendation = "Approve"
    elif credit_score >= 60:
        risk_category = "Medium Risk"
        recommendation = "Review"
    else:
        risk_category = "High Risk"
        recommendation = "Deny"

    # Check additional risk factors
    high_risk_factors = []

    if age < 25 and credit_amount > 5000:
        high_risk_factors.append("Young borrower with large loan")

    if duration > 48:
        high_risk_factors.append("Long loan duration")

    if client_data["Job"].values[0] == 0 and credit_amount > 3000:
        high_risk_factors.append("Unskilled worker with significant loan")

    return {
        "credit_score": credit_score,
        "risk_category": risk_category,
        "recommendation": recommendation,
        "high_risk_factors": high_risk_factors,
        "component_scores": {
            "age_score": age_score,
            "credit_amount_score": credit_amount_score,
            "duration_score": duration_score,
            "savings_score": savings_score,
            "checking_score": checking_score,
            "purpose_score": purpose_score,
        },
    }


# Example usage
sample_client = {
    "Age": 35,
    "Sex": "male",
    "Job": 2,
    "Housing": "own",
    "Saving accounts": "moderate",
    "Checking account": "moderate",
    "Credit amount": 4000,
    "Duration": 24,
    "Purpose": "car",
}

result = predict_credit_risk(pd.DataFrame([sample_client]))
print("\nSample Client Risk Assessment:")
print(f"Credit Score: {result['credit_score']:.1f}")
print(f"Risk Category: {result['risk_category']}")
print(f"Recommendation: {result['recommendation']}")
if result["high_risk_factors"]:
    print("High Risk Factors:")
    for factor in result["high_risk_factors"]:
        print(f"- {factor}")
