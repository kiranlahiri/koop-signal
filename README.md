# koop-signal

Educational use only. Not investment advice. No warranty.


python - << 'PY'
from src.data.loader import fetch_ohlc
from src.strategies.sma_crossover import sma_signal
from src.backtest.engine import run_long_flat
from src.backtest.metrics import sharpe, cagr, max_drawdown

px = fetch_ohlc("AAPL", "2015-01-01", "2025-09-01", auto_adjust=True)
sig = sma_signal(px, 20, 50)
res = run_long_flat(px, sig, cost_bps=1.0)
print("OK  Sharpe:", round(sharpe(res["returns"]), 2),
      "CAGR:", f"{cagr(res['returns']):.2%}",
      "MDD:", f"{max_drawdown(res['equity']):.2%}")
PY
