import numpy as np
import pandas as pd

TRADING_DAYS = 252


def _as_series(x):
    if isinstance(x, pd.DataFrame):
        if x.shape[1] == 1:
            return x.iloc[:, 0]
        raise ValueError("Expected 1-D series; got multi-column DataFrame.")
    return x


def to_returns(prices: pd.Series) -> pd.Series:
    s = _as_series(prices).astype(float)
    return s.pct_change().fillna(0.0)


def cagr(returns, periods_per_year: int = TRADING_DAYS) -> float:
    r = _as_series(returns).astype(float)
    gross = float((1 + r).prod())
    years = len(r) / periods_per_year
    return (gross ** (1 / years) - 1) if years > 0 else 0.0


def sharpe(returns, rf: float = 0.0, periods_per_year: int = TRADING_DAYS) -> float:
    r = _as_series(returns).astype(float)
    xs = r - rf / periods_per_year
    mu = float(xs.mean()) * periods_per_year
    sig = float(xs.std(ddof=1)) * np.sqrt(periods_per_year)
    return mu / sig if sig > 1e-12 else 0.0


def equity_curve(returns) -> pd.Series:
    r = _as_series(returns).astype(float)
    return (1 + r).cumprod()


def max_drawdown(equity) -> float:
    e = _as_series(equity).astype(float)
    peak = e.cummax()
    dd = e / peak - 1.0
    return float(dd.min())
