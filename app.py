import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# PAGE CONFIG & CUSTOM THEME
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="NEXUS Trading Intelligence",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Root palette ── */
:root {
    --bg:        #09090f;
    --surface:   #10101a;
    --border:    #1e1e2e;
    --accent:    #00e5b3;
    --accent2:   #7c6aff;
    --danger:    #ff4d6d;
    --text:      #e2e2f0;
    --muted:     #6b6b8a;
    --mono:      'Space Mono', monospace;
    --sans:      'DM Sans', sans-serif;
}

/* ── Global reset ── */
html, body, [class*="css"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--sans) !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] .stMarkdown h3 {
    font-family: var(--mono) !important;
    color: var(--accent) !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
}

/* ── Headers ── */
h1, h2, h3 { font-family: var(--mono) !important; }
h1 { color: var(--accent) !important; letter-spacing: -0.02em; }
h2 { color: var(--text) !important; font-size: 1rem !important;
     text-transform: uppercase; letter-spacing: 0.12em;
     border-bottom: 1px solid var(--border); padding-bottom: 0.5rem; }

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    padding: 1.2rem !important;
}
[data-testid="stMetricLabel"] {
    font-family: var(--mono) !important;
    font-size: 0.65rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    color: var(--muted) !important;
}
[data-testid="stMetricValue"] {
    font-family: var(--mono) !important;
    font-size: 1.6rem !important;
    color: var(--accent) !important;
}
[data-testid="stMetricDelta"] { font-family: var(--mono) !important; }

/* ── Input fields ── */
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input,
[data-testid="stSelectbox"] select {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    font-family: var(--mono) !important;
    border-radius: 6px !important;
}
[data-testid="stTextInput"] input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(0,229,179,0.15) !important;
}

/* ── Buttons ── */
[data-testid="stButton"] button {
    background: transparent !important;
    border: 1px solid var(--accent) !important;
    color: var(--accent) !important;
    font-family: var(--mono) !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.1em !important;
    border-radius: 4px !important;
    transition: all 0.2s ease !important;
}
[data-testid="stButton"] button:hover {
    background: var(--accent) !important;
    color: var(--bg) !important;
}

/* ── Info / success / warning boxes ── */
[data-testid="stAlert"] {
    background: var(--surface) !important;
    border-radius: 6px !important;
    border-left: 3px solid var(--accent) !important;
    font-family: var(--mono) !important;
    font-size: 0.8rem !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}

/* ── Divider ── */
hr { border-color: var(--border) !important; }

/* ── Tabs ── */
[data-testid="stTabs"] [role="tab"] {
    font-family: var(--mono) !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: var(--muted) !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom: 2px solid var(--accent) !important;
}

/* ── Signal badge ── */
.signal-buy {
    display:inline-block; padding:0.35rem 1rem;
    background: rgba(0,229,179,0.12); border:1px solid var(--accent);
    color:var(--accent); font-family:var(--mono); font-size:0.85rem;
    border-radius:4px; letter-spacing:0.1em;
}
.signal-sell {
    display:inline-block; padding:0.35rem 1rem;
    background: rgba(255,77,109,0.12); border:1px solid var(--danger);
    color:var(--danger); font-family:var(--mono); font-size:0.85rem;
    border-radius:4px; letter-spacing:0.1em;
}
.signal-hold {
    display:inline-block; padding:0.35rem 1rem;
    background: rgba(124,106,255,0.12); border:1px solid var(--accent2);
    color:var(--accent2); font-family:var(--mono); font-size:0.85rem;
    border-radius:4px; letter-spacing:0.1em;
}
.hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 2.2rem;
    font-weight: 700;
    color: #00e5b3;
    letter-spacing: -0.03em;
    line-height: 1.1;
    margin-bottom: 0.2rem;
}
.hero-sub {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.85rem;
    color: #6b6b8a;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.stat-block {
    background: #10101a;
    border: 1px solid #1e1e2e;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# PLOTLY CHART THEME
# ──────────────────────────────────────────────
CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Space Mono, monospace", color="#6b6b8a", size=11),
    xaxis=dict(gridcolor="#1e1e2e", showgrid=True, zeroline=False,
               linecolor="#1e1e2e", tickfont=dict(color="#6b6b8a")),
    yaxis=dict(gridcolor="#1e1e2e", showgrid=True, zeroline=False,
               linecolor="#1e1e2e", tickfont=dict(color="#6b6b8a")),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#1e1e2e",
                borderwidth=1, font=dict(color="#e2e2f0")),
    hovermode="x unified",
    hoverlabel=dict(bgcolor="#10101a", bordercolor="#1e1e2e",
                    font=dict(family="Space Mono, monospace", color="#e2e2f0")),
    margin=dict(l=10, r=10, t=40, b=10),
)

# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⬡ NEXUS Trading Intelligence")
    st.markdown("---")
    st.markdown("### 🎯 Configuration")

    ticker = st.text_input("Stock Ticker", "AAPL").strip().upper()

    period_map = {
        "1 Year": "1y", "2 Years": "2y",
        "3 Years": "3y", "5 Years": "5y",
    }
    period_label = st.selectbox("Lookback Period", list(period_map.keys()), index=3)
    period = period_map[period_label]

    initial_capital = st.number_input(
        "Starting Capital ($)", min_value=1000, max_value=10_000_000,
        value=10_000, step=1000
    )

    st.markdown("---")
    st.markdown("### 🔧 RSI Settings")
    rsi_window    = st.slider("RSI Window",     7, 21, 14)
    rsi_oversold  = st.slider("Oversold Level", 20, 40, 30)
    rsi_overbought = st.slider("Overbought Level", 60, 80, 70)

    st.markdown("---")
    st.markdown("### 📐 Moving Averages")
    ma_short = st.slider("Short MA Period", 5,  30,  10)
    ma_long  = st.slider("Long MA Period",  20, 200, 50)

    st.markdown("---")
    run_btn = st.button("▶  RUN ANALYSIS", use_container_width=True)

# ──────────────────────────────────────────────
# HERO HEADER
# ──────────────────────────────────────────────
st.markdown(
    '<div class="hero-title">⬡ NEXUS TRADING INTELLIGENCE</div>'
    '<div class="hero-sub">Multi-Strategy Backtesting & ML Signal Engine</div>',
    unsafe_allow_html=True,
)
st.markdown("<br>", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# DATA FUNCTIONS
# ──────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def get_data(ticker: str, period: str) -> pd.DataFrame | None:
    try:
        data = yf.download(ticker, period=period, auto_adjust=True, progress=False)
        if data.empty:
            return None
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data = data.dropna()
        if len(data) < 60:
            return None
        return data
    except Exception:
        return None


def calculate_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calculate_bollinger(series: pd.Series, window: int = 20, num_std: float = 2.0):
    sma   = series.rolling(window).mean()
    std   = series.rolling(window).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    return upper, sma, lower


def add_features(data: pd.DataFrame, rsi_window: int, ma_short: int, ma_long: int) -> pd.DataFrame:
    df = data.copy()
    df["RSI"]    = calculate_rsi(df["Close"], rsi_window)
    df[f"MA{ma_short}"] = df["Close"].rolling(ma_short).mean()
    df[f"MA{ma_long}"]  = df["Close"].rolling(ma_long).mean()

    # Bollinger Bands
    df["BB_Upper"], df["BB_Mid"], df["BB_Lower"] = calculate_bollinger(df["Close"])

    # Volume MA
    df["Vol_MA20"] = df["Volume"].rolling(20).mean()

    # Additional ML features
    df["Return_1d"]  = df["Close"].pct_change(1)
    df["Return_5d"]  = df["Close"].pct_change(5)
    df["Volatility"] = df["Return_1d"].rolling(10).std()
    df["MA_Ratio"]   = df[f"MA{ma_short}"] / df[f"MA{ma_long}"]

    # Target: 1 if next-day close > today's close  (label only — not used as a feature)
    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

    return df.dropna()

# ──────────────────────────────────────────────
# PERFORMANCE METRICS
# ──────────────────────────────────────────────
def compute_metrics(equity_curve: pd.Series, initial_capital: float) -> dict:
    """Given a cumulative-profit series (starting at 0), compute key metrics."""
    portfolio = initial_capital + equity_curve

    total_return = (portfolio.iloc[-1] - initial_capital) / initial_capital * 100

    daily_ret = portfolio.pct_change().dropna()
    sharpe = (daily_ret.mean() / daily_ret.std() * np.sqrt(252)
              if daily_ret.std() > 0 else 0.0)

    roll_max  = portfolio.cummax()
    drawdown  = (portfolio - roll_max) / roll_max * 100
    max_dd    = drawdown.min()

    # Win rate from trade-by-trade PnL
    pnl_diff   = equity_curve.diff().dropna()
    trades     = pnl_diff[pnl_diff != 0]
    wins       = (trades > 0).sum()
    total_tr   = len(trades)
    win_rate   = wins / total_tr * 100 if total_tr > 0 else 0.0

    return {
        "Total Return (%)": round(total_return, 2),
        "Sharpe Ratio":      round(sharpe, 2),
        "Max Drawdown (%)":  round(max_dd, 2),
        "Win Rate (%)":      round(win_rate, 2),
        "Total Trades":      total_tr,
        "Final Portfolio ($)": round(portfolio.iloc[-1], 2),
    }

# ──────────────────────────────────────────────
# ML MODEL
# ──────────────────────────────────────────────
FEATURE_COLS = ["RSI", "Return_1d", "Return_5d", "Volatility", "MA_Ratio"]

def train_model(data: pd.DataFrame):
    feature_cols = [c for c in FEATURE_COLS if c in data.columns]
    X = data[feature_cols]
    y = data["Target"]

    if len(X) < 100:
        return None, 0.0, feature_cols

    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=6,
        min_samples_leaf=10,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    return model, acc, feature_cols


# ──────────────────────────────────────────────
# STRATEGIES  (return equity curve as pd.Series)
# ──────────────────────────────────────────────
def _run_equity(data: pd.DataFrame, signals: np.ndarray,
                initial_capital: float) -> pd.Series:
    """
    Convert a buy(1)/sell(0) signal array into a cumulative-profit equity curve.
    Uses a fixed 1-share position. Unrealised P&L at end of data is realised.
    """
    capital   = initial_capital
    buy_price = None
    equity    = []
    profit    = 0.0

    for i, (price, sig) in enumerate(zip(data["Close"].values, signals)):
        if sig == 1 and buy_price is None:
            buy_price = price
        elif sig == 0 and buy_price is not None:
            profit   += price - buy_price
            buy_price = None
        equity.append(profit)

    # Close any open position at last price
    if buy_price is not None:
        profit += data["Close"].iloc[-1] - buy_price

    return pd.Series(equity, index=data.index)


def rsi_strategy(data: pd.DataFrame, oversold: int, overbought: int,
                 initial_capital: float):
    signals    = np.zeros(len(data), dtype=int)
    in_trade   = False
    buy_prices = []
    sell_prices = []
    buy_idx    = []
    sell_idx   = []

    for i in range(len(data)):
        rsi = data["RSI"].iloc[i]
        if rsi < oversold and not in_trade:
            signals[i] = 1
            in_trade   = True
            buy_prices.append(data["Close"].iloc[i])
            buy_idx.append(data.index[i])
        elif rsi > overbought and in_trade:
            signals[i] = 0
            in_trade   = False
            sell_prices.append(data["Close"].iloc[i])
            sell_idx.append(data.index[i])
        else:
            signals[i] = 1 if in_trade else 0

    equity = _run_equity(data, signals, initial_capital)
    return equity, buy_idx, sell_idx, buy_prices, sell_prices


def ml_strategy(data: pd.DataFrame, model, feature_cols: list,
                initial_capital: float):
    if model is None:
        return pd.Series(np.zeros(len(data)), index=data.index), [], [], [], []
    preds = model.predict(data[feature_cols])
    equity = _run_equity(data, preds, initial_capital)
    # Derive signal transitions for trade markers
    buy_idx, sell_idx, buy_p, sell_p = [], [], [], []
    in_trade = False
    for i, (sig, price) in enumerate(zip(preds, data["Close"].values)):
        if sig == 1 and not in_trade:
            buy_idx.append(data.index[i]); buy_p.append(price); in_trade = True
        elif sig == 0 and in_trade:
            sell_idx.append(data.index[i]); sell_p.append(price); in_trade = False
    return equity, buy_idx, sell_idx, buy_p, sell_p


def hybrid_strategy(data: pd.DataFrame, model, feature_cols: list,
                    oversold: int, overbought: int, initial_capital: float):
    if model is None:
        return pd.Series(np.zeros(len(data)), index=data.index), [], [], [], []
    ml_preds = model.predict(data[feature_cols])
    signals  = np.zeros(len(data), dtype=int)
    in_trade = False
    buy_idx, sell_idx, buy_p, sell_p = [], [], [], []

    for i in range(len(data)):
        rsi = data["RSI"].iloc[i]
        ml  = ml_preds[i]
        price = data["Close"].iloc[i]

        if rsi < oversold and ml == 1 and not in_trade:
            signals[i] = 1; in_trade = True
            buy_idx.append(data.index[i]); buy_p.append(price)
        elif (rsi > overbought or ml == 0) and in_trade:
            signals[i] = 0; in_trade = False
            sell_idx.append(data.index[i]); sell_p.append(price)
        else:
            signals[i] = 1 if in_trade else 0

    equity = _run_equity(data, signals, initial_capital)
    return equity, buy_idx, sell_idx, buy_p, sell_p


# ──────────────────────────────────────────────
# CURRENT SIGNAL
# ──────────────────────────────────────────────
def get_current_signal(data: pd.DataFrame, model, feature_cols: list,
                       oversold: int, overbought: int) -> str:
    last  = data.iloc[-1]
    rsi   = last["RSI"]
    ml_sig = None

    if model is not None:
        ml_sig = model.predict(data[feature_cols].iloc[[-1]])[0]

    if rsi < oversold and (ml_sig is None or ml_sig == 1):
        return "BUY"
    elif rsi > overbought and (ml_sig is None or ml_sig == 0):
        return "SELL"
    else:
        return "HOLD"

# ──────────────────────────────────────────────
# CHART HELPERS
# ──────────────────────────────────────────────
def make_candlestick(data, ma_short, ma_long,
                     buy_idx=None, sell_idx=None,
                     buy_prices=None, sell_prices=None,
                     ticker=""):
    ma_s_col = f"MA{ma_short}"
    ma_l_col = f"MA{ma_long}"

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.75, 0.25],
        vertical_spacing=0.02,
        subplot_titles=("", "Volume"),
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data["Open"], high=data["High"],
        low=data["Low"],   close=data["Close"],
        name=ticker,
        increasing_line_color="#00e5b3",
        decreasing_line_color="#ff4d6d",
        increasing_fillcolor="rgba(0,229,179,0.7)",
        decreasing_fillcolor="rgba(255,77,109,0.7)",
    ), row=1, col=1)

    # MAs
    fig.add_trace(go.Scatter(
        x=data.index, y=data[ma_s_col],
        name=f"MA{ma_short}", line=dict(color="#7c6aff", width=1.2),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=data.index, y=data[ma_l_col],
        name=f"MA{ma_long}", line=dict(color="#f9a03f", width=1.2),
    ), row=1, col=1)

    # Bollinger Bands
    fig.add_trace(go.Scatter(
        x=data.index, y=data["BB_Upper"],
        name="BB Upper", line=dict(color="rgba(255,255,255,0.15)", width=1),
        showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=data.index, y=data["BB_Lower"],
        name="BB Lower",
        line=dict(color="rgba(255,255,255,0.15)", width=1),
        fill="tonexty", fillcolor="rgba(255,255,255,0.03)",
        showlegend=False,
    ), row=1, col=1)

    # Buy signals
    if buy_idx and buy_prices:
        fig.add_trace(go.Scatter(
            x=buy_idx, y=buy_prices,
            mode="markers", name="Buy",
            marker=dict(symbol="triangle-up", color="#00e5b3", size=10,
                        line=dict(color="#fff", width=0.5)),
        ), row=1, col=1)

    # Sell signals
    if sell_idx and sell_prices:
        fig.add_trace(go.Scatter(
            x=sell_idx, y=sell_prices,
            mode="markers", name="Sell",
            marker=dict(symbol="triangle-down", color="#ff4d6d", size=10,
                        line=dict(color="#fff", width=0.5)),
        ), row=1, col=1)

    # Volume bars
    colors = ["#00e5b3" if c >= o else "#ff4d6d"
              for c, o in zip(data["Close"], data["Open"])]
    fig.add_trace(go.Bar(
        x=data.index, y=data["Volume"],
        name="Volume", marker_color=colors, opacity=0.6,
    ), row=2, col=1)

    # Volume MA
    fig.add_trace(go.Scatter(
        x=data.index, y=data["Vol_MA20"],
        name="Vol MA20", line=dict(color="#7c6aff", width=1),
    ), row=2, col=1)

    fig.update_layout(**CHART_LAYOUT, height=550,
                      title=dict(text=f"<b>{ticker}</b> Price Chart",
                                 font=dict(family="Space Mono", color="#e2e2f0", size=13)),
                      xaxis_rangeslider_visible=False)
    fig.update_yaxes(title_text="Price ($)", row=1, col=1,
                     title_font=dict(color="#6b6b8a"))
    fig.update_yaxes(title_text="Volume",   row=2, col=1,
                     title_font=dict(color="#6b6b8a"))
    return fig


def make_rsi_chart(data, oversold, overbought):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data.index, y=data["RSI"],
        name="RSI", line=dict(color="#7c6aff", width=1.5),
        fill="tozeroy", fillcolor="rgba(124,106,255,0.06)",
    ))
    fig.add_hrect(y0=overbought, y1=100,
                  fillcolor="rgba(255,77,109,0.07)", line_width=0)
    fig.add_hrect(y0=0, y1=oversold,
                  fillcolor="rgba(0,229,179,0.07)", line_width=0)
    fig.add_hline(y=overbought, line_dash="dash",
                  line_color="rgba(255,77,109,0.6)", line_width=1)
    fig.add_hline(y=oversold,   line_dash="dash",
                  line_color="rgba(0,229,179,0.6)",  line_width=1)
    fig.add_hline(y=50, line_dash="dot",
                  line_color="rgba(255,255,255,0.1)", line_width=1)
    rsi_layout = {**CHART_LAYOUT, "height": 250,
                  "title": dict(text="RSI Oscillator",
                                font=dict(family="Space Mono", color="#e2e2f0", size=13)),
                  "yaxis": dict(**CHART_LAYOUT["yaxis"], range=[0, 100])}
    fig.update_layout(**rsi_layout)
    return fig


def make_equity_chart(curves: dict, initial_capital: float):
    fig = go.Figure()
    colors = {"RSI": "#00e5b3", "ML": "#7c6aff", "Hybrid": "#f9a03f"}
    for name, eq in curves.items():
        portfolio = initial_capital + eq
        fig.add_trace(go.Scatter(
            x=eq.index, y=portfolio,
            name=name, line=dict(color=colors.get(name, "#fff"), width=1.8),
        ))
    fig.add_hline(y=initial_capital, line_dash="dot",
                  line_color="rgba(255,255,255,0.2)", line_width=1,
                  annotation_text="Capital", annotation_font_color="#6b6b8a")
    fig.update_layout(**CHART_LAYOUT, height=350,
                      title=dict(text="Portfolio Value Over Time",
                                 font=dict(family="Space Mono", color="#e2e2f0", size=13)),
                      yaxis_title="Portfolio ($)")
    return fig


def make_drawdown_chart(curves: dict, initial_capital: float):
    fig = go.Figure()
    colors = {"RSI": "#00e5b3", "ML": "#7c6aff", "Hybrid": "#f9a03f"}
    for name, eq in curves.items():
        portfolio = initial_capital + eq
        roll_max  = portfolio.cummax()
        drawdown  = (portfolio - roll_max) / roll_max * 100
        fig.add_trace(go.Scatter(
            x=eq.index, y=drawdown,
            name=name, line=dict(color=colors.get(name, "#fff"), width=1.5),
            fill="tozeroy", fillcolor=f"rgba{tuple(int(colors.get(name,'#fff').lstrip('#')[i:i+2],16) for i in (0,2,4)) + (0.06,)}",
        ))
    fig.update_layout(**CHART_LAYOUT, height=280,
                      title=dict(text="Drawdown (%)",
                                 font=dict(family="Space Mono", color="#e2e2f0", size=13)))
    return fig

# ──────────────────────────────────────────────
# MAIN LOGIC
# ──────────────────────────────────────────────
if not ticker:
    st.info("Enter a ticker symbol in the sidebar to begin.")
    st.stop()

with st.spinner(f"Fetching {ticker} data…"):
    raw_data = get_data(ticker, period)

if raw_data is None:
    st.error(
        f"❌  Could not retrieve data for **{ticker}**. "
        "Please check the ticker symbol and try again."
    )
    st.stop()

with st.spinner("Engineering features…"):
    data = add_features(raw_data, rsi_window, ma_short, ma_long)

with st.spinner("Training ML model…"):
    model, acc, feature_cols = train_model(data)

with st.spinner("Running strategies…"):
    rsi_eq, rsi_bi, rsi_si, rsi_bp, rsi_sp = rsi_strategy(
        data, rsi_oversold, rsi_overbought, initial_capital)
    ml_eq,  ml_bi,  ml_si,  ml_bp,  ml_sp  = ml_strategy(
        data, model, feature_cols, initial_capital)
    hyb_eq, hyb_bi, hyb_si, hyb_bp, hyb_sp = hybrid_strategy(
        data, model, feature_cols, rsi_oversold, rsi_overbought, initial_capital)

rsi_metrics = compute_metrics(rsi_eq, initial_capital)
ml_metrics  = compute_metrics(ml_eq,  initial_capital)
hyb_metrics = compute_metrics(hyb_eq, initial_capital)

# Best strategy by total return
all_returns = {
    "RSI":    rsi_metrics["Total Return (%)"],
    "ML":     ml_metrics["Total Return (%)"],
    "Hybrid": hyb_metrics["Total Return (%)"],
}
best_name = max(all_returns, key=all_returns.get)

# Current live signal
current_signal = get_current_signal(
    data, model, feature_cols, rsi_oversold, rsi_overbought
)

# ──────────────────────────────────────────────
# CURRENT SIGNAL + BEST STRATEGY BANNER
# ──────────────────────────────────────────────
col_sig, col_best, col_acc, col_rsi_now = st.columns(4)

with col_sig:
    badge_cls = {"BUY": "signal-buy", "SELL": "signal-sell", "HOLD": "signal-hold"}[current_signal]
    st.markdown("**TODAY'S SIGNAL**")
    st.markdown(f'<span class="{badge_cls}">▶ {current_signal}</span>', unsafe_allow_html=True)

with col_best:
    st.markdown("**BEST STRATEGY**")
    st.markdown(f'<span class="signal-buy">🏆 {best_name}</span>', unsafe_allow_html=True)

with col_acc:
    st.metric("ML Accuracy", f"{round(acc*100,1)}%" if model else "N/A")

with col_rsi_now:
    st.metric("Current RSI", f"{round(data['RSI'].iloc[-1], 1)}")

st.markdown("<br>", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# TABS
# ──────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Performance", "📈 Charts", "〰 Indicators",
    "🤖 ML Insights", "📋 Data"
])

# ─── TAB 1: Performance ───────────────────────
with tab1:
    st.markdown("## Strategy Performance")

    def render_strategy_card(name, metrics, color_var):
        cols = st.columns(3)
        cols[0].metric("Total Return",    f"{metrics['Total Return (%)']:+.2f}%")
        cols[1].metric("Sharpe Ratio",    metrics["Sharpe Ratio"])
        cols[2].metric("Max Drawdown",    f"{metrics['Max Drawdown (%)']:.2f}%")
        cols2 = st.columns(3)
        cols2[0].metric("Win Rate",       f"{metrics['Win Rate (%)']:.1f}%")
        cols2[1].metric("Total Trades",   metrics["Total Trades"])
        cols2[2].metric("Final Portfolio",f"${metrics['Final Portfolio ($)']:,.0f}")

    rsi_tab, ml_tab, hyb_tab = st.tabs(["RSI Strategy", "ML Strategy", "Hybrid Strategy"])
    with rsi_tab:
        render_strategy_card("RSI", rsi_metrics, "#00e5b3")
    with ml_tab:
        render_strategy_card("ML",  ml_metrics,  "#7c6aff")
    with hyb_tab:
        render_strategy_card("Hybrid", hyb_metrics, "#f9a03f")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("## Equity Curves")
    st.plotly_chart(
        make_equity_chart({"RSI": rsi_eq, "ML": ml_eq, "Hybrid": hyb_eq}, initial_capital),
        use_container_width=True
    )

    st.markdown("## Drawdown Analysis")
    st.plotly_chart(
        make_drawdown_chart({"RSI": rsi_eq, "ML": ml_eq, "Hybrid": hyb_eq}, initial_capital),
        use_container_width=True
    )

    # Comparison table
    st.markdown("## Side-by-Side Comparison")
    comp_df = pd.DataFrame({
        "RSI":    rsi_metrics,
        "ML":     ml_metrics,
        "Hybrid": hyb_metrics,
    }).T
    st.dataframe(comp_df.style.highlight_max(axis=0, color="#1a3a2a")
                              .highlight_min(axis=0, color="#3a1a1a"),
                 use_container_width=True)

    # Export
    csv = comp_df.to_csv().encode("utf-8")
    st.download_button(
        "⬇  Export Performance CSV", data=csv,
        file_name=f"{ticker}_strategy_comparison.csv",
        mime="text/csv",
    )

# ─── TAB 2: Charts ────────────────────────────
with tab2:
    st.markdown("## Price Chart")

    strategy_for_chart = st.selectbox(
        "Show signals for strategy",
        ["RSI", "ML", "Hybrid"],
        key="chart_strategy"
    )

    signal_map = {
        "RSI":    (rsi_bi, rsi_si, rsi_bp, rsi_sp),
        "ML":     (ml_bi,  ml_si,  ml_bp,  ml_sp),
        "Hybrid": (hyb_bi, hyb_si, hyb_bp, hyb_sp),
    }
    bi, si, bp, sp = signal_map[strategy_for_chart]

    st.plotly_chart(
        make_candlestick(data, ma_short, ma_long, bi, si, bp, sp, ticker),
        use_container_width=True
    )

# ─── TAB 3: Indicators ────────────────────────
with tab3:
    st.markdown("## RSI Oscillator")
    st.plotly_chart(make_rsi_chart(data, rsi_oversold, rsi_overbought),
                    use_container_width=True)

    st.markdown("## Bollinger Bands")
    bb_fig = go.Figure()
    bb_fig.add_trace(go.Scatter(
        x=data.index, y=data["BB_Upper"],
        name="Upper Band", line=dict(color="rgba(255,77,109,0.5)", width=1),
    ))
    bb_fig.add_trace(go.Scatter(
        x=data.index, y=data["BB_Lower"],
        name="Lower Band",
        line=dict(color="rgba(0,229,179,0.5)", width=1),
        fill="tonexty", fillcolor="rgba(255,255,255,0.03)",
    ))
    bb_fig.add_trace(go.Scatter(
        x=data.index, y=data["BB_Mid"],
        name="Middle Band", line=dict(color="rgba(255,255,255,0.3)", width=1, dash="dot"),
    ))
    bb_fig.add_trace(go.Scatter(
        x=data.index, y=data["Close"],
        name="Close", line=dict(color="#7c6aff", width=1.5),
    ))
    bb_fig.update_layout(**CHART_LAYOUT, height=300,
                         title=dict(text="Bollinger Bands",
                                    font=dict(family="Space Mono", color="#e2e2f0", size=13)))
    st.plotly_chart(bb_fig, use_container_width=True)

# ─── TAB 4: ML Insights ───────────────────────
with tab4:
    st.markdown("## ML Model Insights")
    if model is None:
        st.warning("Insufficient data to train the ML model (need ≥ 100 rows after feature engineering).")
    else:
        st.markdown(f"""
<div class="stat-block">
<b>Model:</b> RandomForestClassifier &nbsp;|&nbsp;
<b>Estimators:</b> 150 &nbsp;|&nbsp;
<b>Max Depth:</b> 6 &nbsp;|&nbsp;
<b>Test Accuracy:</b> {round(acc*100,1)}%
</div>
""", unsafe_allow_html=True)

        # Feature importance
        importances = model.feature_importances_
        feat_df = pd.DataFrame({
            "Feature":    feature_cols,
            "Importance": importances,
        }).sort_values("Importance", ascending=True)

        fi_fig = go.Figure(go.Bar(
            x=feat_df["Importance"], y=feat_df["Feature"],
            orientation="h",
            marker=dict(
                color=feat_df["Importance"],
                colorscale=[[0, "#1e1e2e"], [1, "#00e5b3"]],
                showscale=False,
            ),
        ))
        fi_fig.update_layout(**CHART_LAYOUT, height=280,
                             title=dict(text="Feature Importance",
                                        font=dict(family="Space Mono", color="#e2e2f0", size=13)))
        st.plotly_chart(fi_fig, use_container_width=True)

        # Prediction confidence on last 60 days
        st.markdown("## Prediction Probability (Last 60 Days)")
        recent = data.iloc[-60:].copy()
        proba  = model.predict_proba(recent[feature_cols])[:, 1]
        prob_fig = go.Figure()
        prob_fig.add_trace(go.Scatter(
            x=recent.index, y=proba,
            name="P(Up)", line=dict(color="#7c6aff", width=1.5),
            fill="tozeroy", fillcolor="rgba(124,106,255,0.08)",
        ))
        prob_fig.add_hline(y=0.5, line_dash="dash",
                           line_color="rgba(255,255,255,0.2)", line_width=1)
        prob_layout = {**CHART_LAYOUT, "height": 280,
                       "yaxis": dict(**CHART_LAYOUT["yaxis"], range=[0, 1]),
                       "title": dict(text="Predicted Probability of Upward Move",
                                     font=dict(family="Space Mono", color="#e2e2f0", size=13))}
        prob_fig.update_layout(**prob_layout)
        st.plotly_chart(prob_fig, use_container_width=True)

        # Disclaimer
        st.caption(
            "⚠️ The ML model is trained on historical data with an 80/20 time-based split. "
            "Accuracy reflects out-of-sample test performance. Past performance does not "
            "guarantee future results. This is not financial advice."
        )

# ─── TAB 5: Data ──────────────────────────────
with tab5:
    st.markdown("## Raw Data Preview (Last 30 Rows)")
    display_cols = ["Open", "High", "Low", "Close", "Volume",
                    "RSI", f"MA{ma_short}", f"MA{ma_long}",
                    "BB_Upper", "BB_Lower", "Return_1d"]
    display_cols = [c for c in display_cols if c in data.columns]
    st.dataframe(
        data[display_cols].tail(30).style.format("{:.2f}"),
        use_container_width=True,
    )

    raw_csv = data[display_cols].to_csv().encode("utf-8")
    st.download_button(
        "⬇  Export Full Data CSV", data=raw_csv,
        file_name=f"{ticker}_data.csv", mime="text/csv",
    )

# ──────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<p style="font-family:Space Mono,monospace;font-size:0.65rem;'
    'color:#6b6b8a;text-align:center;letter-spacing:0.1em;">'
    'NEXUS TRADING INTELLIGENCE &nbsp;·&nbsp; FOR EDUCATIONAL USE ONLY &nbsp;·&nbsp; '
    'NOT FINANCIAL ADVICE &nbsp;·&nbsp; DATA VIA YAHOO FINANCE'
    '</p>',
    unsafe_allow_html=True,
)