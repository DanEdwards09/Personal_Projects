"""
Customer Risk & Pricing Model for Insurance Underwriting
========================================================

This model predicts claims likelihood and policy risk using classification models
to inform underwriting and pricing decisions in the insurance sector.

"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")


class InsuranceRiskModel:
    """
    A comprehensive insurance risk assessment model that predicts claim likelihood
    and calculates risk-adjusted pricing for insurance policies.
    """

    def __init__(self):
        self.models = {
            "logistic_regression": LogisticRegression(random_state=42),
            "random_forest": RandomForestClassifier(random_state=42),
        }
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.best_model = None
        self.feature_importance = None

    def generate_sample_data(self, n_samples=5000):
        """Generate realistic insurance customer data for demonstration"""
        np.random.seed(42)

        # Customer demographics
        age = np.random.normal(45, 15, n_samples).clip(18, 80)
        gender = np.random.choice(["M", "F"], n_samples)

        # Policy details
        policy_type = np.random.choice(
            ["Auto", "Home", "Life", "Health"], n_samples, p=[0.4, 0.25, 0.15, 0.2]
        )
        coverage_amount = np.random.lognormal(10, 1, n_samples).clip(10000, 1000000)
        deductible = np.random.choice([250, 500, 1000, 2500], n_samples)

        # Risk factors
        credit_score = np.random.normal(650, 100, n_samples).clip(300, 850)
        previous_claims = np.random.poisson(0.5, n_samples).clip(0, 5)
        years_with_company = np.random.exponential(3, n_samples).clip(0, 20)

        # Location risk (simplified)
        location_risk = np.random.choice(
            ["Low", "Medium", "High"], n_samples, p=[0.5, 0.35, 0.15]
        )

        # Create claim probability based on risk factors
        risk_score = (
            (age < 25) * 0.3  # Young drivers higher risk
            + (age > 65) * 0.2  # Elderly higher risk
            + (credit_score < 600) * 0.4  # Poor credit
            + (previous_claims > 0) * 0.5  # Previous claims history
            + (location_risk == "High") * 0.3
            + (coverage_amount > 100000) * 0.1
            + np.random.normal(0, 0.1, n_samples)  # Random noise
        )

        # Generate claims based on risk score
        claim_probability = 1 / (1 + np.exp(-risk_score))
        has_claim = np.random.binomial(1, claim_probability, n_samples)

        # Create DataFrame
        data = pd.DataFrame(
            {
                "age": age.round(0).astype(int),
                "gender": gender,
                "policy_type": policy_type,
                "coverage_amount": coverage_amount.round(2),
                "deductible": deductible,
                "credit_score": credit_score.round(0).astype(int),
                "previous_claims": previous_claims,
                "years_with_company": years_with_company.round(1),
                "location_risk": location_risk,
                "has_claim": has_claim,
            }
        )

        return data

    def preprocess_data(self, data):
        """Preprocess the data for modeling"""
        df = data.copy()

        # Encode categorical variables
        categorical_cols = ["gender", "policy_type", "location_risk"]

        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df[col] = self.label_encoders[col].fit_transform(df[col])
            else:
                df[col] = self.label_encoders[col].transform(df[col])

        # Create additional features
        df["coverage_per_year"] = df["coverage_amount"] / (df["years_with_company"] + 1)
        df["age_group"] = pd.cut(
            df["age"],
            bins=[0, 25, 35, 50, 65, 100],
            labels=["Young", "Adult", "Middle", "Senior", "Elderly"],
        )
        df["age_group"] = LabelEncoder().fit_transform(df["age_group"])

        return df

    def train_models(self, X_train, y_train):
        """Train multiple models and select the best one"""
        results = {}

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)

        for name, model in self.models.items():
            # Hyperparameter tuning
            if name == "logistic_regression":
                param_grid = {
                    "C": [0.1, 1, 10],
                    "penalty": ["l1", "l2"],
                    "solver": ["liblinear"],
                }
            else:  # random_forest
                param_grid = {
                    "n_estimators": [100, 200],
                    "max_depth": [5, 10, None],
                    "min_samples_split": [2, 5],
                }

            grid_search = GridSearchCV(
                model, param_grid, cv=5, scoring="roc_auc", n_jobs=-1
            )

            if name == "logistic_regression":
                grid_search.fit(X_train_scaled, y_train)
            else:
                grid_search.fit(X_train, y_train)

            # Cross-validation scores
            if name == "logistic_regression":
                cv_scores = cross_val_score(
                    grid_search.best_estimator_,
                    X_train_scaled,
                    y_train,
                    cv=5,
                    scoring="roc_auc",
                )
            else:
                cv_scores = cross_val_score(
                    grid_search.best_estimator_,
                    X_train,
                    y_train,
                    cv=5,
                    scoring="roc_auc",
                )

            results[name] = {
                "model": grid_search.best_estimator_,
                "best_params": grid_search.best_params_,
                "cv_auc_mean": cv_scores.mean(),
                "cv_auc_std": cv_scores.std(),
            }

        # Select best model
        best_model_name = max(results.keys(), key=lambda x: results[x]["cv_auc_mean"])
        self.best_model = results[best_model_name]["model"]
        self.best_model_name = best_model_name

        print(f"Best model: {best_model_name}")
        print(f"Best parameters: {results[best_model_name]['best_params']}")
        print(
            f"Cross-validation AUC: {results[best_model_name]['cv_auc_mean']:.4f} ± {results[best_model_name]['cv_auc_std']:.4f}"
        )

        return results

    def evaluate_model(self, X_test, y_test, feature_names):
        """Evaluate the best model on test data"""
        if self.best_model_name == "logistic_regression":
            X_test_processed = self.scaler.transform(X_test)
        else:
            X_test_processed = X_test

        # Predictions
        y_pred = self.best_model.predict(X_test_processed)
        y_pred_proba = self.best_model.predict_proba(X_test_processed)[:, 1]

        # Calculate metrics
        auc_score = roc_auc_score(y_test, y_pred_proba)

        print(f"\nModel Evaluation Results:")
        print(f"AUC Score: {auc_score:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        # Feature importance
        if hasattr(self.best_model, "feature_importances_"):
            self.feature_importance = pd.DataFrame(
                {
                    "feature": feature_names,
                    "importance": self.best_model.feature_importances_,
                }
            ).sort_values("importance", ascending=False)

        return {
            "auc_score": auc_score,
            "y_pred": y_pred,
            "y_pred_proba": y_pred_proba,
            "classification_report": classification_report(
                y_test, y_pred, output_dict=True
            ),
        }

    def calculate_risk_premium(self, claim_probability, base_premium=1000):
        """Calculate risk-adjusted premium based on claim probability"""
        # Risk multiplier based on claim probability
        risk_multiplier = 1 + (claim_probability * 2)  # Max 3x base premium

        # Add administrative costs and profit margin
        admin_margin = 0.2  # 20%

        risk_premium = base_premium * risk_multiplier * (1 + admin_margin)

        return risk_premium

    def generate_pricing_recommendations(self, customer_data):
        """Generate pricing recommendations for new customers"""
        # Preprocess customer data
        processed_data = self.preprocess_data(customer_data)

        # Remove target variable if present
        if "has_claim" in processed_data.columns:
            X = processed_data.drop("has_claim", axis=1)
        else:
            X = processed_data

        # Make predictions
        if self.best_model_name == "logistic_regression":
            X_processed = self.scaler.transform(X)
        else:
            X_processed = X

        claim_probabilities = self.best_model.predict_proba(X_processed)[:, 1]

        # Calculate risk-adjusted premiums
        premiums = [self.calculate_risk_premium(prob) for prob in claim_probabilities]

        # Risk categories
        risk_categories = [
            "Low Risk" if p < 0.3 else "Medium Risk" if p < 0.6 else "High Risk"
            for p in claim_probabilities
        ]

        recommendations = pd.DataFrame(
            {
                "claim_probability": claim_probabilities,
                "recommended_premium": premiums,
                "risk_category": risk_categories,
            }
        )

        return recommendations

    def plot_model_performance(self, X_test, y_test):
        """Plot model performance visualizations"""
        if self.best_model_name == "logistic_regression":
            X_test_processed = self.scaler.transform(X_test)
        else:
            X_test_processed = X_test

        y_pred_proba = self.best_model.predict_proba(X_test_processed)[:, 1]

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc_score = roc_auc_score(y_test, y_pred_proba)

        axes[0, 0].plot(fpr, tpr, label=f"ROC Curve (AUC = {auc_score:.4f})")
        axes[0, 0].plot([0, 1], [0, 1], "k--", label="Random")
        axes[0, 0].set_xlabel("False Positive Rate")
        axes[0, 0].set_ylabel("True Positive Rate")
        axes[0, 0].set_title("ROC Curve")
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Prediction Distribution
        axes[0, 1].hist(y_pred_proba[y_test == 0], alpha=0.7, label="No Claim", bins=30)
        axes[0, 1].hist(y_pred_proba[y_test == 1], alpha=0.7, label="Claim", bins=30)
        axes[0, 1].set_xlabel("Predicted Probability")
        axes[0, 1].set_ylabel("Frequency")
        axes[0, 1].set_title("Prediction Distribution")
        axes[0, 1].legend()

        # Feature Importance (if available)
        if self.feature_importance is not None:
            top_features = self.feature_importance.head(10)
            axes[1, 0].barh(range(len(top_features)), top_features["importance"])
            axes[1, 0].set_yticks(range(len(top_features)))
            axes[1, 0].set_yticklabels(top_features["feature"])
            axes[1, 0].set_xlabel("Importance")
            axes[1, 0].set_title("Top 10 Feature Importances")

        # Risk Distribution
        risk_buckets = pd.cut(y_pred_proba, bins=10)
        risk_analysis = pd.DataFrame(
            {
                "predicted_prob": y_pred_proba,
                "actual_claim": y_test,
                "risk_bucket": risk_buckets,
            }
        )

        bucket_stats = (
            risk_analysis.groupby("risk_bucket")
            .agg({"predicted_prob": "mean", "actual_claim": "mean"})
            .reset_index()
        )

        axes[1, 1].plot(
            bucket_stats["predicted_prob"],
            bucket_stats["actual_claim"],
            "o-",
            markersize=8,
        )
        axes[1, 1].plot([0, 1], [0, 1], "r--", alpha=0.7, label="Perfect Calibration")
        axes[1, 1].set_xlabel("Mean Predicted Probability")
        axes[1, 1].set_ylabel("Actual Claim Rate")
        axes[1, 1].set_title("Model Calibration")
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        plt.tight_layout()
        plt.show()


def main():
    """Main execution function"""
    print("Insurance Customer Risk & Pricing Model")
    print("=====================================")

    # Initialize model
    risk_model = InsuranceRiskModel()

    # Generate sample data
    print("\n1. Generating sample insurance data...")
    data = risk_model.generate_sample_data(5000)
    print(f"Generated {len(data)} customer records")
    print(f"Claim rate: {data['has_claim'].mean():.2%}")

    # Display sample data
    print("\nSample data:")
    print(data.head())

    # Preprocess data
    print("\n2. Preprocessing data...")
    processed_data = risk_model.preprocess_data(data)

    # Prepare features and target
    X = processed_data.drop("has_claim", axis=1)
    y = processed_data["has_claim"]
    feature_names = X.columns.tolist()

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")

    # Train models
    print("\n3. Training and tuning models...")
    training_results = risk_model.train_models(X_train, y_train)

    # Evaluate model
    print("\n4. Evaluating model performance...")
    evaluation_results = risk_model.evaluate_model(X_test, y_test, feature_names)

    # Generate pricing recommendations for sample customers
    print("\n5. Generating pricing recommendations...")
    sample_customers = data.sample(5).drop("has_claim", axis=1)
    recommendations = risk_model.generate_pricing_recommendations(sample_customers)

    print("\nSample Pricing Recommendations:")
    for i, (idx, customer) in enumerate(sample_customers.iterrows()):
        rec = recommendations.iloc[i]
        print(f"\nCustomer {i + 1}:")
        print(f"  Age: {customer['age']}, Policy: {customer['policy_type']}")
        print(
            f"  Credit Score: {customer['credit_score']}, Previous Claims: {customer['previous_claims']}"
        )
        print(f"  Claim Probability: {rec['claim_probability']:.2%}")
        print(f"  Risk Category: {rec['risk_category']}")
        print(f"  Recommended Premium: ${rec['recommended_premium']:.2f}")

    # Plot performance
    print("\n6. Generating performance visualizations...")
    risk_model.plot_model_performance(X_test, y_test)

    # Summary statistics
    print("\n7. Model Summary:")
    print(f"Final AUC Score: {evaluation_results['auc_score']:.4f}")
    print(
        f"Model achieves the target AUC > 0.82: {evaluation_results['auc_score'] > 0.82}"
    )

    if risk_model.feature_importance is not None:
        print("\nTop 5 Risk Factors:")
        for i, row in risk_model.feature_importance.head(5).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")


if __name__ == "__main__":
    main()
