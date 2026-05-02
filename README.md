## Overview
 
The financial trading sector generates enormous volumes of data daily, yet most users lack efficient tools to analyze it in a structured way. Nexus Trading Dashboard addresses this by combining exploratory data analysis, technical indicators, and machine learning into a single interactive web application. Users can backtest three distinct strategies — RSI-based, ML-based, and a Hybrid of both — against any publicly traded stock, compare performance metrics, and get a live trading signal for the current day.
 
---
 
## Features
 
- **Multi-Strategy Backtesting** — RSI, Random Forest ML, and Hybrid strategies run simultaneously and are compared side by side
- **Live Signal Engine** — Generates a BUY / SELL / HOLD signal for the current day based on RSI and ML predictions
- **Technical Indicators** — RSI oscillator, Moving Averages (configurable short/long), Bollinger Bands, and Volume MA
- **ML Insights** — Feature importance chart and prediction probability over the last 60 days
- **Interactive Charts** — Candlestick charts with trade markers, equity curves, and drawdown analysis powered by Plotly
- **Configurable Parameters** — Adjust ticker, lookback period, starting capital, RSI settings, and MA periods from the sidebar
- **Data Export** — Download strategy comparison and raw data as CSV
---
 
## Strategies
 
### RSI Strategy
Buys when RSI falls below the oversold threshold and sells when RSI rises above the overbought threshold. Simple momentum-reversal logic.
 
### ML Strategy (Random Forest Classifier)
Trains a `RandomForestClassifier` on historical features (RSI, 1-day return, 5-day return, volatility, MA ratio) with an 80/20 time-based train/test split to predict whether the next day's close will be higher.
 
### Hybrid Strategy
Requires both the RSI oversold condition **and** an ML buy prediction to enter a trade. Exits when RSI becomes overbought **or** ML predicts down. More conservative, fewer signals.
 
---
 
## Performance Metrics
 
Each strategy is evaluated on:
 
| Metric | Description |
|---|---|
| Total Return (%) | Net profit as a percentage of starting capital |
| Sharpe Ratio | Risk-adjusted return (annualized) |
| Max Drawdown (%) | Worst peak-to-trough loss |
| Win Rate (%) | Percentage of profitable trades |
| Total Trades | Number of completed round-trip trades |
| Final Portfolio ($) | Ending portfolio value |
 
---
 
## Tech Stack
 
| Layer | Technology |
|---|---|
| Dashboard | Streamlit |
| Data Fetching | yfinance (Yahoo Finance) |
| Data Manipulation | Pandas, NumPy |
| Machine Learning | scikit-learn (RandomForestClassifier) |
| Visualization | Plotly |
| Language | Python 3 |
 
---
 
## Getting Started
 
### Prerequisites
 
```
python >= 3.9
```
 
### Installation
 
```bash
# Clone the repository
git clone <your-repo-url>
cd nexus-trading-dashboard
 
# Install dependencies
pip install streamlit pandas numpy yfinance scikit-learn plotly
```
 
### Running Locally
 
```bash
streamlit run app.py
```
 
The app opens at `http://localhost:8501`.
 
---
 
## Usage
 
1. Enter a **Stock Ticker** in the sidebar (e.g., `AAPL`, `TSLA`, `INFY.NS`)
2. Select a **Lookback Period** (1–5 years)
3. Set your **Starting Capital**
4. Adjust **RSI Settings** and **Moving Average** periods as desired
5. Click **▶ RUN ANALYSIS**
6. Explore the five tabs: Performance · Charts · Indicators · ML Insights · Data
---
 
## Project Structure
 
```
nexus-trading-dashboard/
└── app.py        # Complete single-file Streamlit application
                  # Contains: data fetching, feature engineering,
                  #           strategy logic, ML model, chart builders,
                  #           and full dashboard UI
```
 
---
 
## Key Implementation Details
 
**Feature Engineering (`add_features`)** — Computes RSI, short/long MAs, Bollinger Bands, Volume MA, 1-day and 5-day returns, rolling volatility, and an MA ratio. The `Target` column (next-day direction) is used as the ML label and is excluded from features.
 
**Equity Curve (`_run_equity`)** — Simulates a fixed 1-share position. Tracks cumulative profit from trade-by-trade PnL and realises any open position at the last available price.
 
**Caching** — `@st.cache_data(ttl=300)` caches fetched data for 5 minutes to avoid redundant API calls during reruns.
 
---
 
## Disclaimer
 
⚠️ This project is built for **educational purposes only**. The ML model is trained on historical data; past performance does not guarantee future results. This is **not financial advice**. Data is sourced from Yahoo Finance.
 
---
 
