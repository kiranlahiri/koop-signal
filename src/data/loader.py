import pandas as pd
import yfinance as yf


def fetch_ohlc(
    ticker: str, start: str, end: str, interval: str = "1d", auto_adjust: bool = True
) -> pd.Series:
    # Explicitly set auto_adjust so the behavior is clear and the warning disappears
    df = yf.download(
        ticker,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=auto_adjust,
        progress=False,
    )
    if df.empty or "Close" not in df:
        raise ValueError(f"No data for {ticker} in [{start}, {end}]")

    s = df["Close"].dropna().copy()
    s.name = ticker  # <-- set the Series name directly
    return s
