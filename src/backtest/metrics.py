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


# newer metrics


def sortino(returns, rf: float = 0.0, periods_per_year: int = TRADING_DAYS) -> float:
    r = _as_series(returns).astype(float) - rf / periods_per_year
    downside = r[r < 0.0]
    dvol = float(downside.std(ddof=1)) * np.sqrt(periods_per_year)
    mu = float(r.mean()) * periods_per_year
    return mu / dvol if dvol > 1e-12 else 0.0


def calmar(returns) -> float:
    r = _as_series(returns).astype(float)
    eq = equity_curve(r)
    mdd = abs(max_drawdown(eq))
    g = cagr(r)
    return g / mdd if mdd > 1e-12 else 0.0


def time_under_water(equity) -> int:
    """Longest consecutive run of days when equity < running peak."""
    e = _as_series(equity).astype(float)
    uw = e < e.cummax()
    if not uw.any():
        return 0
    groups = (uw != uw.shift()).cumsum()
    run_lengths = uw.groupby(groups).sum()  # sums True in each run
    return int(run_lengths.max())


def turnover(positions):
    """Returns (daily_turnover, annualized_turnover). For {-1..1} or {0,1} positions."""
    p = _as_series(positions).fillna(0.0).clip(-1, 1)
    tau = p.diff().abs().fillna(0.0)  # -1->1 flip counts as 2
    daily = float(tau.mean())
    return daily, daily * TRADING_DAYS


def summarize(returns, equity, positions) -> dict:
    """Collect key stats into a JSON-friendly dict."""
    r = _as_series(returns)
    e = _as_series(equity)
    pos = _as_series(positions)

    daily_to = turnover(pos)
    out = {
        "N_days": int(len(r)),
        "AnnVol": float(r.std(ddof=1)) * np.sqrt(TRADING_DAYS),
        "Sharpe": sharpe(r),
        "Sortino": sortino(r),
        "CAGR": cagr(r),
        "MaxDD": max_drawdown(e),
        "Calmar": calmar(r),
        "TuW_days": time_under_water(e),
        "Turnover_daily": daily_to[0],
        "Turnover_annual": daily_to[1],
    }
    # force plain floats for JSON
    return {
        k: (float(v) if isinstance(v, (np.floating, pd.Series)) else v)
        for k, v in out.items()
    }
