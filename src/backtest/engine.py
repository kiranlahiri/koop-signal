# src/backtest/engine.py
import pandas as pd


def _as_series(x):
    if isinstance(x, pd.DataFrame):
        if x.shape[1] == 1:
            return x.iloc[:, 0]
        raise ValueError(f"Expected Series, got DataFrame with {x.shape[1]} columns.")
    return x


def run_long_flat(close: pd.Series, signal: pd.Series, cost_bps: float = 1.0):
    """
    Long/flat backtest with next-bar execution and cost in bps on position change.
    Returns dict of Series: returns, equity, positions, trades.
    """
    # ensure 1-D
    close = _as_series(close).astype(float)
    signal = _as_series(signal).astype(float)

    # daily returns from prices
    rets = close.pct_change().fillna(0.0)

    # trade on NEXT bar (no look-ahead)
    pos = signal.shift(1).fillna(0.0).clip(-1, 1)

    # a "trade" occurs when position changes
    trades = (pos != pos.shift(1)).fillna(False).astype(int)

    # cost charged only on trade days
    cost = trades * (cost_bps / 1e4)

    strat_rets = pos * rets - cost
    equity = (1.0 + strat_rets).cumprod()

    return {
        "returns": strat_rets,
        "equity": equity,
        "positions": pos,
        "trades": trades,
    }
