# Crude Oil Price Forecasting & Risk Analysis

**Goal:** Forecast crude oil prices (WTI/Brent) and estimate market risk using Value-at-Risk (VaR).  
**Why it matters:** Directly relevant to commercial trading, risk, and supply/scheduling at companies like 

## 🔧 Tech Stack
- Python (pandas, numpy, matplotlib, scikit-learn)
- Time series: statsmodels (ARIMA/SARIMAX), Prophet *(optional)*
- ML: RandomForestRegressor; LSTM *(optional, via TensorFlow)*
- Data: Yahoo Finance (yfinance) or EIA API (optional)
- Dashboard: Streamlit *(optional quick win)*

## 🗂 Project Structure
```
crude_oil_forecasting_var/
├── crude_forecast_var.ipynb       # Main notebook (run end-to-end)
├── requirements.txt               # Minimal dependencies
└── README.md                      # This file
```
*(Files are generated for you in this download bundle.)*

## 🚀 How to Run
1. **Create a virtual environment** (recommended).
2. Install deps:
   ```bash
   pip install -r requirements.txt
   ```
3. Launch the notebook:
   ```bash
   jupyter lab  # or jupyter notebook
   ```

## 📈 Deliverables
- Clean EDA with volatility/rolling views and annotated market shocks (COVID 2020, Ukraine 2022).
- Multiple forecasting approaches (Naïve, ARIMA/SARIMAX, Prophet*, Random Forest, LSTM*).
- Evaluation (RMSE/MAE) + visual comparison.
- Risk analysis with Historical Simulation VaR (95%, 99%) + rolling VaR.
- Exported `predictions.csv`, `metrics.json`, and plots under `artifacts/`.

\* *Optional components gracefully skip if not installed.*


## 🛡 Ethics & Limitations
- Educational project; **not** investment advice.
- Forecasts on highly stochastic commodities have high uncertainty, especially around structural breaks.