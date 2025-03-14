import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
)

# Add at the beginning of your code
import warnings

warnings.filterwarnings(
    "ignore", category=RuntimeWarning, message="invalid value encountered in greater"
)

# 1. DATA COLLECTION - using the S&P 500 dataset from Kaggle
dataset_path = "/kaggle/input/sandp500/all_stocks_5yr.csv"

# Create dictionaries to store data and results
stock_data = {}
models = {}
results = {}

# Load the dataset
print(f"Loading dataset from {dataset_path}...")
try:
    all_stocks_df = pd.read_csv(dataset_path)
    print(
        f"Successfully loaded dataset with {all_stocks_df.shape[0]} rows and {all_stocks_df.shape[1]} columns"
    )
    print(f"Available columns: {all_stocks_df.columns.tolist()}")
    print(f"First few rows:\n{all_stocks_df.head(3)}")

    # Check available tickers
    available_tickers = all_stocks_df["Name"].unique()
    print(f"Number of unique tickers: {len(available_tickers)}")
    print(f"Sample of available tickers: {available_tickers[:10]}")

except Exception as e:
    print(f"Error loading dataset: {e}")
    # If there's an error, we'll exit the code
    import sys

    sys.exit("Dataset loading failed")

# Choose tickers we want to analyze
tickers = ["SPY", "AAPL", "MSFT"]

# Check which of our tickers are actually in the dataset
available_tickers_set = set(available_tickers)
selected_tickers = [ticker for ticker in tickers if ticker in available_tickers_set]

if not selected_tickers:
    print(
        f"None of the requested tickers {tickers} found in dataset. Selecting some available ones instead."
    )
    # Select a few popular stocks if our preferred ones aren't available
    fallback_tickers = ["AAPL", "MSFT", "AMZN", "GOOGL", "FB", "BRK.B", "JNJ", "JPM"]
    selected_tickers = [
        ticker for ticker in fallback_tickers if ticker in available_tickers_set
    ][:3]

    if not selected_tickers:
        # If still none found, just take the first 3 from the dataset
        selected_tickers = list(available_tickers[:3])

print(f"Selected tickers for analysis: {selected_tickers}")

# Process each ticker
for ticker in selected_tickers:
    print(f"\nProcessing {ticker}...")

    # Filter data for this ticker
    ticker_data = all_stocks_df[all_stocks_df["Name"] == ticker].copy()

    # Make sure we have data
    if ticker_data.empty:
        print(f"No data available for {ticker}. Skipping...")
        continue

    # Convert date column to datetime and set as index
    ticker_data["date"] = pd.to_datetime(ticker_data["date"])
    ticker_data.set_index("date", inplace=True)

    # Sort by date (just to be safe)
    ticker_data = ticker_data.sort_index()

    # Print basic info
    print(f"Data ranges from {ticker_data.index.min()} to {ticker_data.index.max()}")
    print(f"Data shape: {ticker_data.shape}")

    # Rename columns to standard format if needed
    # Check if the column names match what we expect
    column_mapping = {}
    if "open" in ticker_data.columns:
        column_mapping["open"] = "Open"
    if "high" in ticker_data.columns:
        column_mapping["high"] = "High"
    if "low" in ticker_data.columns:
        column_mapping["low"] = "Low"
    if "close" in ticker_data.columns:
        column_mapping["close"] = "Close"
    if "volume" in ticker_data.columns:
        column_mapping["volume"] = "Volume"

    if column_mapping:
        ticker_data.rename(columns=column_mapping, inplace=True)

    # Feature engineering
    # 1. Add moving averages
    ticker_data["MA5"] = ticker_data["Close"].rolling(window=5).mean()
    ticker_data["MA20"] = ticker_data["Close"].rolling(window=20).mean()

    # 2. Add price momentum
    ticker_data["Return_1d"] = ticker_data["Close"].pct_change()
    ticker_data["Return_5d"] = ticker_data["Close"].pct_change(periods=5)

    # 3. Add volume features
    ticker_data["Volume_Change"] = ticker_data["Volume"].pct_change()

    # 4. Add target variable (1 if price goes up tomorrow, 0 otherwise)
    ticker_data["Target"] = (
        ticker_data["Close"].shift(-1) > ticker_data["Close"]
    ).astype(int)

    # Drop NaN values
    ticker_data = ticker_data.dropna()

    # Store processed data
    stock_data[ticker] = ticker_data

    # Print summary of processed data
    print(f"Processed data shape: {ticker_data.shape}")
    print(
        f"First few rows of processed data:\n{ticker_data[['Close', 'MA5', 'MA20', 'Target']].head(3)}"
    )

# 2. MODEL TRAINING
for ticker in stock_data.keys():
    print(f"\nTraining model for {ticker}...")
    df = stock_data[ticker]

    # Prepare features and target
    features = ["MA5", "MA20", "Return_1d", "Return_5d", "Volume_Change"]
    X = df[features]
    y = df["Target"]

    # Split data (80% train, 20% test)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    print(f"Training data size: {X_train.shape}")
    print(f"Test data size: {X_test.shape}")

    # Train a simple XGBoost model
    model = XGBClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42
    )

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    predictions = model.predict(X_test)
    pred_proba = model.predict_proba(X_test)[:, 1]

    # Calculate accuracy
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy for {ticker}: {accuracy:.4f}")

    # Store model and results
    models[ticker] = model
    results[ticker] = {
        "accuracy": accuracy,
        "confusion_matrix": confusion_matrix(y_test, predictions),
        "classification_report": classification_report(
            y_test, predictions, output_dict=True
        ),
        "feature_importance": pd.DataFrame(
            {"feature": features, "importance": model.feature_importances_}
        ).sort_values("importance", ascending=False),
        "test_data": df.iloc[train_size:].copy(),
        "predictions": predictions,
        "pred_proba": pred_proba,
    }

    # Add predictions to test data
    results[ticker]["test_data"]["Predicted"] = predictions
    results[ticker]["test_data"]["Predicted_Proba"] = pred_proba
    results[ticker]["test_data"]["Correct"] = (
        results[ticker]["test_data"]["Predicted"]
        == results[ticker]["test_data"]["Target"]
    ).astype(int)

# 3. VISUALIZATION
for ticker in models.keys():
    print(f"\nVisualizing results for {ticker}...")
    result = results[ticker]

    # Create a figure with 2x2 subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Confusion Matrix
    sns.heatmap(
        result["confusion_matrix"], annot=True, fmt="d", cmap="Blues", ax=axs[0, 0]
    )
    axs[0, 0].set_title(f"{ticker} Confusion Matrix")
    axs[0, 0].set_xlabel("Predicted")
    axs[0, 0].set_ylabel("Actual")

    # 2. Feature Importance
    sns.barplot(
        x="importance", y="feature", data=result["feature_importance"], ax=axs[0, 1]
    )
    axs[0, 1].set_title(f"{ticker} Feature Importance")

    # 3. ROC Curve
    fpr, tpr, _ = roc_curve(result["test_data"]["Target"], result["pred_proba"])
    roc_auc = auc(fpr, tpr)
    axs[1, 0].plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    axs[1, 0].plot([0, 1], [0, 1], "k--")
    axs[1, 0].set_xlabel("False Positive Rate")
    axs[1, 0].set_ylabel("True Positive Rate")
    axs[1, 0].set_title(f"{ticker} ROC Curve")
    axs[1, 0].legend()

    # 4. Price and Predictions
    test_data = result["test_data"]
    axs[1, 1].plot(test_data.index, test_data["Close"], label="Stock Price")

    # Green dots for correct predictions
    correct_predictions = test_data[test_data["Correct"] == 1]
    axs[1, 1].scatter(
        correct_predictions.index,
        correct_predictions["Close"],
        color="green",
        label="Correct Prediction",
        alpha=0.5,
    )

    # Red dots for incorrect predictions
    incorrect_predictions = test_data[test_data["Correct"] == 0]
    axs[1, 1].scatter(
        incorrect_predictions.index,
        incorrect_predictions["Close"],
        color="red",
        label="Incorrect Prediction",
        alpha=0.5,
    )

    axs[1, 1].set_title(f"{ticker} Prediction Performance")
    axs[1, 1].legend()

    plt.tight_layout()
    plt.suptitle(f"{ticker} Analysis - Accuracy: {result['accuracy']:.4f}", fontsize=16)
    plt.subplots_adjust(top=0.93)
    plt.show()

# 4. SIMPLE TRADING STRATEGY SIMULATION
for ticker in models.keys():
    print(f"\nBacktesting strategy for {ticker}...")
    test_data = results[ticker]["test_data"].copy()

    # Initialize portfolio
    initial_capital = 10000
    cash = initial_capital
    shares = 0
    portfolio_values = [initial_capital]

    # Set threshold for taking action (confidence level)
    threshold = 0.55

    # Track trades
    trades = []

    for i in range(1, len(test_data)):
        current_price = test_data["Close"].iloc[i - 1]
        next_price = test_data["Close"].iloc[i]
        pred_proba = test_data["Predicted_Proba"].iloc[i - 1]
        date = test_data.index[i]

        # High confidence in UP prediction - Buy
        if pred_proba > threshold and shares == 0:
            shares = cash / current_price
            cash = 0
            trades.append(
                {
                    "Date": date,
                    "Action": "BUY",
                    "Price": current_price,
                    "Shares": shares,
                    "Portfolio Value": shares * current_price,
                }
            )

        # High confidence in DOWN prediction - Sell
        elif pred_proba < (1 - threshold) and shares > 0:
            cash = shares * current_price
            trades.append(
                {
                    "Date": date,
                    "Action": "SELL",
                    "Price": current_price,
                    "Shares": shares,
                    "Portfolio Value": cash,
                }
            )
            shares = 0

        # Update portfolio value
        portfolio_values.append(cash + (shares * next_price))

    # Calculate returns
    portfolio_returns = pd.Series(portfolio_values).pct_change().fillna(0)
    cumulative_returns = (1 + portfolio_returns).cumprod() - 1

    # Calculate buy and hold returns
    buy_hold_returns = (test_data["Close"] / test_data["Close"].iloc[0]) - 1

    # Plot strategy performance
    plt.figure(figsize=(12, 6))
    plt.plot(test_data.index, cumulative_returns * 100, label="Strategy Returns (%)")
    plt.plot(test_data.index, buy_hold_returns * 100, label="Buy & Hold Returns (%)")

    # Add buy/sell markers if there were any trades
    if trades:
        trades_df = pd.DataFrame(trades)
        buy_points = trades_df[trades_df["Action"] == "BUY"]
        sell_points = trades_df[trades_df["Action"] == "SELL"]

        # Map dates to corresponding index in cumulative_returns
        for idx, row in buy_points.iterrows():
            date_idx = test_data.index.get_loc(row["Date"])
            plt.scatter(
                row["Date"],
                cumulative_returns.iloc[date_idx] * 100,
                color="green",
                marker="^",
                s=100,
                label="Buy" if idx == 0 else "",
            )

        for idx, row in sell_points.iterrows():
            date_idx = test_data.index.get_loc(row["Date"])
            plt.scatter(
                row["Date"],
                cumulative_returns.iloc[date_idx] * 100,
                color="red",
                marker="v",
                s=100,
                label="Sell" if idx == 0 else "",
            )

    plt.title(f"{ticker} Trading Strategy vs Buy & Hold")
    plt.xlabel("Date")
    plt.ylabel("Returns (%)")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Calculate strategy metrics
    final_return = cumulative_returns.iloc[-1] * 100
    buy_hold_return = buy_hold_returns.iloc[-1] * 100

    print(f"\n{ticker} Trading Strategy Results:")
    print(f"Strategy Return: {final_return:.2f}%")
    print(f"Buy & Hold Return: {buy_hold_return:.2f}%")
    print(f"Outperformance: {final_return - buy_hold_return:.2f}%")

    if trades:
        print(f"Number of trades: {len(trades)}")
        print(f"Number of buys: {len(buy_points)}")
        print(f"Number of sells: {len(sell_points)}")

# 5. FINAL SUMMARY
print("\n=== PROJECT SUMMARY ===")
print("Model Performance:")
for ticker in models.keys():
    print(f"{ticker}: Accuracy = {results[ticker]['accuracy']:.4f}")

# Find best performing stock based on model accuracy
if len(models) > 0:
    best_model = max(models.keys(), key=lambda x: results[x]["accuracy"])
    print(
        f"\nBest performing model: {best_model} with {results[best_model]['accuracy']:.4f} accuracy"
    )

    # Top features across all models
    print("\nTop features by importance:")
    for ticker in models.keys():
        top_feature = results[ticker]["feature_importance"]["feature"].iloc[0]
        print(f"{ticker}: {top_feature}")
else:
    print("No models were successfully trained.")

# 6. PROJECT EXPLANATION
print("""
\n=== PROJECT EXPLANATION ===

This stock market prediction project uses XGBoost to predict daily price movements:

1. DATA: We used historical stock data from the S&P 500 dataset, focusing on key stocks.

2. FEATURES: We engineered technical indicators including:
   - Moving averages (MA5, MA20)
   - Price returns (1-day and 5-day)
   - Volume changes

3. MODEL: We trained an XGBoost classifier to predict whether prices would go up or down
   the next trading day.

4. EVALUATION: We assessed model performance using:
   - Accuracy metrics
   - Confusion matrices
   - ROC curves and AUC values
   - Feature importance analysis

5. TRADING STRATEGY: We implemented a simple algorithmic trading strategy:
   - Buy when model predicts >55% probability of price increase
   - Sell when model predicts >55% probability of price decrease
   - Compared returns against a buy-and-hold baseline

Future improvements could include:
- Adding more sophisticated features (RSI, MACD, Bollinger Bands)
- Hyperparameter tuning for better model performance
- More advanced trading strategies with position sizing
- Ensemble models combining multiple prediction approaches
""")
