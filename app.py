import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from src.data.loader import fetch_ohlc
from src.strategies.sma_crossover import sma_signal
from src.backtest.engine import run_long_flat
from src.backtest.metrics import summarize, equity_curve

st.title("KOOP-SIGNAL: SMA Backtest Demo")

# Sidebar inputs
ticker = st.sidebar.text_input("Ticker", "AAPL")
start = st.sidebar.date_input("Start", pd.to_datetime("2015-01-01"))
end = st.sidebar.date_input("End", pd.to_datetime("2025-09-01"))
fast = st.sidebar.number_input("Fast SMA", min_value=5, max_value=100, value=20)
slow = st.sidebar.number_input("Slow SMA", min_value=10, max_value=200, value=50)
cost_bps = st.sidebar.number_input("Cost (bps per trade)", min_value=0.0, value=1.0)
auto_adjust = st.sidebar.checkbox("Auto-adjust prices", value=True)

# Data + backtest
close = fetch_ohlc(ticker, str(start), str(end), auto_adjust=auto_adjust)
signal = sma_signal(close, fast, slow)
res = run_long_flat(close, signal, cost_bps=cost_bps)

# Benchmark
spy = fetch_ohlc("SPY", str(start), str(end), auto_adjust=auto_adjust)
bench_ret = spy.pct_change().reindex(close.index).fillna(0.0)
bench_eq = equity_curve(bench_ret)

# Metrics
# Benchmark (SPY)
spy = fetch_ohlc("SPY", str(start), str(end), auto_adjust=auto_adjust)
bench_ret = spy.pct_change().reindex(close.index).fillna(0.0).squeeze()
bench_eq = equity_curve(bench_ret)

# Metrics
metrics = summarize(res["returns"], res["equity"], signal.shift(1).fillna(0.0))
metrics_df = pd.DataFrame([metrics]).T.rename(columns={0: "Strategy"})

# Add benchmark (SPY) stats
try:
    sharpe_spy = float(bench_ret.mean()) / float(bench_ret.std()) * (252**0.5)
except ZeroDivisionError:
    sharpe_spy = 0.0

cagr_spy = (1 + bench_ret).prod() ** (252 / len(bench_ret)) - 1
maxdd_spy = (bench_eq / bench_eq.cummax() - 1).min()

metrics_df["SPY"] = [
    None,  # N days
    None,  # AnnVol
    f"{sharpe_spy:.2f}",  # Sharpe
    None,  # Sortino
    f"{cagr_spy:.2%}",  # CAGR
    f"{maxdd_spy:.2%}",  # MaxDD
    None,
    None,
    None,
    None,  # filler rows
]

st.subheader("Performance Metrics")
st.dataframe(metrics_df)

# Plot equity curves
st.subheader("Equity Curves")
fig, ax = plt.subplots(figsize=(8, 4))
res["equity"].plot(ax=ax, label="Strategy")
bench_eq.plot(ax=ax, label="SPY")
ax.legend()
st.pyplot(fig)
