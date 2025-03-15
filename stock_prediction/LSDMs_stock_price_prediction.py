import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    LSTM,
    Dropout,
    BatchNormalization,
    Bidirectional,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Check for GPU availability
print("GPU Available: ", tf.config.list_physical_devices("GPU"))

# 1. DATA COLLECTION - using the S&P 500 dataset
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

    # Check available tickers
    available_tickers = all_stocks_df["Name"].unique()
    print(f"Number of unique tickers: {len(available_tickers)}")

except Exception as e:
    print(f"Error loading dataset: {e}")
    import sys

    sys.exit("Dataset loading failed")

# Choose tickers to analyze
tickers = ["AAPL", "MSFT"]  # Focus on these since we already have results for them
selected_tickers = [ticker for ticker in tickers if ticker in available_tickers]

print(f"Selected tickers for analysis: {selected_tickers}")


# 2. ENHANCED DATA PREPARATION FUNCTION
def prepare_enhanced_lstm_data(
    dataframe, sequence_length=20, use_technical_indicators=True
):
    """
    Prepare data for LSTM model with additional features:
    1. Add technical indicators as features
    2. Scale all features
    3. Create sequences of specified length
    4. Split into X (features) and y (target)
    """
    # Create a copy of dataframe to avoid modifying the original
    df = dataframe.copy()

    # Add technical indicators if requested
    if use_technical_indicators:
        # Moving averages
        df["MA5"] = df["Close"].rolling(window=5).mean()
        df["MA10"] = df["Close"].rolling(window=10).mean()
        df["MA20"] = df["Close"].rolling(window=20).mean()

        # Price momentum and volatility
        df["Return_1d"] = df["Close"].pct_change()
        df["Return_5d"] = df["Close"].pct_change(periods=5)
        df["Volatility"] = df["Return_1d"].rolling(window=10).std()

        # Relative Strength Index (RSI)
        delta = df["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df["RSI"] = 100 - (100 / (1 + rs))

        # Bollinger Bands
        df["BB_Middle"] = df["Close"].rolling(window=20).mean()
        df["BB_Std"] = df["Close"].rolling(window=20).std()
        df["BB_Upper"] = df["BB_Middle"] + 2 * df["BB_Std"]
        df["BB_Lower"] = df["BB_Middle"] - 2 * df["BB_Std"]
        df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / df["BB_Middle"]

        # Price relative to Bollinger Bands
        df["BB_Position"] = (df["Close"] - df["BB_Lower"]) / (
            df["BB_Upper"] - df["BB_Lower"]
        )

        # Volume indicators
        df["Volume_Change"] = df["Volume"].pct_change()
        df["Volume_MA5"] = df["Volume"].rolling(window=5).mean()
        df["Volume_Ratio"] = df["Volume"] / df["Volume_MA5"]

    # Drop NaN values
    df = df.dropna()

    # Features to use
    if use_technical_indicators:
        # Use price, volume and technical indicators
        feature_columns = [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "MA5",
            "MA10",
            "MA20",
            "Return_1d",
            "Return_5d",
            "Volatility",
            "RSI",
            "BB_Position",
            "BB_Width",
            "Volume_Change",
            "Volume_Ratio",
        ]
    else:
        # Use only price and volume
        feature_columns = ["Open", "High", "Low", "Close", "Volume"]

    # Create target: 1 if price goes up, 0 if down
    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

    # Remove last row (since it doesn't have a target)
    df = df.iloc[:-1]

    # Extract features and target
    data = df[feature_columns].values
    targets = df["Target"].values

    # Scale the features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # Create sequences
    X = []
    y = []

    for i in range(len(scaled_data) - sequence_length):
        X.append(scaled_data[i : i + sequence_length])
        y.append(targets[i + sequence_length])

    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)

    # Get the dates corresponding to each sequence (for backtesting)
    sequence_dates = df.index[sequence_length : len(X) + sequence_length]

    return X, y, scaler, sequence_dates, feature_columns


# 3. Process each ticker
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

    # Sort by date
    ticker_data = ticker_data.sort_index()

    # Rename columns to standard format if needed
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

    # Save the original data for later
    stock_data[ticker] = ticker_data
    print(f"Data shape for {ticker}: {ticker_data.shape}")

# 4. BUILD AND TRAIN ENHANCED LSTM MODELS
for ticker in stock_data.keys():
    print(f"\nTraining enhanced LSTM model for {ticker}...")

    # Get the data
    df = stock_data[ticker]

    # Define sequence length (lookback period)
    sequence_length = 40  # Increased from 10 to 40 days for better pattern recognition

    # Prepare data for LSTM with technical indicators
    X, y, scaler, sequence_dates, feature_columns = prepare_enhanced_lstm_data(
        df, sequence_length=sequence_length, use_technical_indicators=True
    )

    # Make sure we have enough data
    if len(X) < 50:
        print(f"Not enough data for {ticker} after sequence creation. Skipping...")
        continue

    # Split the data (80% train, 20% test)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    dates_test = sequence_dates[train_size:]

    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Number of features: {X_train.shape[2]}")

    # Build the enhanced LSTM model
    model = Sequential(
        [
            # First Bidirectional LSTM layer
            Bidirectional(
                LSTM(100, return_sequences=True),
                input_shape=(sequence_length, X_train.shape[2]),
            ),
            BatchNormalization(),
            Dropout(0.3),
            # Second Bidirectional LSTM layer
            Bidirectional(LSTM(100, return_sequences=True)),
            BatchNormalization(),
            Dropout(0.3),
            # Third LSTM layer
            LSTM(100, return_sequences=False),
            BatchNormalization(),
            Dropout(0.3),
            # Dense layers
            Dense(50, activation="relu"),
            BatchNormalization(),
            Dropout(0.3),
            # Output layer
            Dense(1, activation="sigmoid"),
        ]
    )

    # Compile the model with a lower learning rate
    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    # Set up callbacks
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=15, restore_best_weights=True, verbose=1
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=5, min_lr=0.00001, verbose=1
    )

    # Print model summary
    model.summary()

    # Train the model with more epochs and callbacks
    history = model.fit(
        X_train,
        y_train,
        epochs=100,  # More epochs, but with early stopping
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping, reduce_lr],
        verbose=1,
        shuffle=False,  # Important for time series data
    )

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy for {ticker}: {accuracy:.4f}")

    # Make predictions
    pred_proba = model.predict(X_test)
    predictions = (pred_proba > 0.5).astype(int)

    # Store the model and results
    models[ticker] = model

    # Create a DataFrame for the test data with predictions
    test_data = pd.DataFrame(
        {
            "Close": df["Close"].loc[dates_test].values,
            "Target": y_test,
            "Predicted": predictions.flatten(),
            "Predicted_Proba": pred_proba.flatten(),
            "Correct": (predictions.flatten() == y_test).astype(int),
        },
        index=dates_test,
    )

    # Store results
    results[ticker] = {
        "accuracy": accuracy,
        "confusion_matrix": confusion_matrix(y_test, predictions),
        "classification_report": classification_report(
            y_test, predictions, output_dict=True
        ),
        "test_data": test_data,
        "history": history.history,
        "feature_columns": feature_columns,
        "sequence_length": sequence_length,
    }

    # Plot training history
    plt.figure(figsize=(12, 5))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title(f"{ticker} - Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title(f"{ticker} - Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()

# 5. VISUALIZATION
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

    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(
        result["test_data"]["Target"], result["test_data"]["Predicted_Proba"]
    )
    roc_auc = auc(fpr, tpr)
    axs[0, 1].plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    axs[0, 1].plot([0, 1], [0, 1], "k--")
    axs[0, 1].set_xlabel("False Positive Rate")
    axs[0, 1].set_ylabel("True Positive Rate")
    axs[0, 1].set_title(f"{ticker} ROC Curve")
    axs[0, 1].legend()

    # 3. Price and Predictions
    test_data = result["test_data"]
    axs[1, 0].plot(test_data.index, test_data["Close"], label="Stock Price")

    # Green dots for correct predictions
    correct_predictions = test_data[test_data["Correct"] == 1]
    axs[1, 0].scatter(
        correct_predictions.index,
        correct_predictions["Close"],
        color="green",
        label="Correct Prediction",
        alpha=0.5,
    )

    # Red dots for incorrect predictions
    incorrect_predictions = test_data[test_data["Correct"] == 0]
    axs[1, 0].scatter(
        incorrect_predictions.index,
        incorrect_predictions["Close"],
        color="red",
        label="Incorrect Prediction",
        alpha=0.5,
    )

    axs[1, 0].set_title(f"{ticker} Prediction Performance")
    axs[1, 0].legend()

    # 4. Prediction Distribution
    sns.histplot(
        result["test_data"]["Predicted_Proba"], bins=20, kde=True, ax=axs[1, 1]
    )
    axs[1, 1].axvline(0.5, color="red", linestyle="--")
    axs[1, 1].set_title("Prediction Probability Distribution")
    axs[1, 1].set_xlabel("Predicted Probability of Price Increase")

    plt.tight_layout()
    plt.suptitle(
        f"{ticker} Enhanced LSTM Analysis - Accuracy: {result['accuracy']:.4f}",
        fontsize=16,
    )
    plt.subplots_adjust(top=0.93)
    plt.show()

# 6. OPTIMIZED TRADING STRATEGY SIMULATION
for ticker in models.keys():
    print(f"\nBacktesting enhanced strategy for {ticker}...")
    test_data = results[ticker]["test_data"].copy()

    # Initialize portfolio with various confidence thresholds to find optimal value
    thresholds = [0.5, 0.55, 0.6, 0.65, 0.7]
    best_threshold = 0.5
    best_return = -float("inf")
    best_portfolio_values = []

    for threshold in thresholds:
        # Initialize portfolio
        initial_capital = 10000
        cash = initial_capital
        shares = 0
        portfolio_values = [initial_capital]

        for i in range(1, len(test_data)):
            current_price = test_data["Close"].iloc[i - 1]
            next_price = test_data["Close"].iloc[i]
            pred_proba = test_data["Predicted_Proba"].iloc[i - 1]

            # High confidence in UP prediction - Buy
            if pred_proba > threshold and shares == 0:
                shares = cash / current_price
                cash = 0

            # High confidence in DOWN prediction - Sell
            elif pred_proba < (1 - threshold) and shares > 0:
                cash = shares * current_price
                shares = 0

            # Update portfolio value
            portfolio_values.append(cash + (shares * next_price))

        # Calculate returns
        final_return = (portfolio_values[-1] / initial_capital - 1) * 100

        # Check if this threshold gives better returns
        if final_return > best_return:
            best_return = final_return
            best_threshold = threshold
            best_portfolio_values = portfolio_values.copy()

    print(f"Optimal confidence threshold for {ticker}: {best_threshold}")

    # Re-run the strategy with the best threshold for visualization
    initial_capital = 10000
    cash = initial_capital
    shares = 0
    portfolio_values = [initial_capital]
    trades = []

    for i in range(1, len(test_data)):
        current_price = test_data["Close"].iloc[i - 1]
        next_price = test_data["Close"].iloc[i]
        pred_proba = test_data["Predicted_Proba"].iloc[i - 1]
        date = test_data.index[i]

        # High confidence in UP prediction - Buy
        if pred_proba > best_threshold and shares == 0:
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
        elif pred_proba < (1 - best_threshold) and shares > 0:
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

    plt.title(
        f"{ticker} Enhanced LSTM Trading Strategy vs Buy & Hold (Threshold: {best_threshold})"
    )
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

# 7. FINAL SUMMARY
print("\n=== PROJECT SUMMARY ===")
print("Enhanced Model Performance:")
for ticker in models.keys():
    print(f"{ticker}: Accuracy = {results[ticker]['accuracy']:.4f}")

# Compare with previous results
print("\nComparing with previous results:")
print("Previous AAPL Accuracy: 0.4760")
print("Previous MSFT Accuracy: 0.4480")

# Find best performing stock based on model accuracy
if len(models) > 0:
    best_model = max(models.keys(), key=lambda x: results[x]["accuracy"])
    print(
        f"\nBest performing model: {best_model} with {results[best_model]['accuracy']:.4f} accuracy"
    )
else:
    print("No models were successfully trained.")
