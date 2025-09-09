import pandas as pd


def sma_signal(close: pd.Series, fast: int = 20, slow: int = 50) -> pd.Series:
    close = close.astype(float)
    f = close.rolling(fast, min_periods=fast).mean()
    s = close.rolling(slow, min_periods=slow).mean()
    return (f > s).astype(float)  # Series
