# scripts/results.py
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.data.loader import fetch_ohlc
from src.strategies.sma_crossover import sma_signal
from src.backtest.engine import run_long_flat
from src.backtest.metrics import (
    summarize,
    sharpe,
    sortino,
    cagr,
    max_drawdown,
    equity_curve,
)


def _to_series(x, index=None):
    """Ensure x is a 1-D pandas Series with the given index (if provided)."""
    if isinstance(x, pd.DataFrame):
        if x.shape[1] == 1:
            x = x.iloc[:, 0]
        else:
            raise ValueError(f"Expected 1-D; got DataFrame with {x.shape[1]} cols")
    elif isinstance(x, np.ndarray):
        x = pd.Series(x.ravel(), index=index)  # flatten (N,1) -> (N,)
    return x


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ticker", required=True)
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--fast", type=int, default=20)
    p.add_argument("--slow", type=int, default=50)
    p.add_argument("--cost_bps", type=float, default=1.0)
    p.add_argument("--auto_adjust", action="store_true")
    p.add_argument("--outdir", default="assets")
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Strategy series
    close = fetch_ohlc(args.ticker, args.start, args.end, auto_adjust=args.auto_adjust)
    signal = sma_signal(close, args.fast, args.slow)
    res = run_long_flat(close, signal, cost_bps=args.cost_bps)

    # Benchmark: SPY buy & hold
    spy_close = fetch_ohlc("SPY", args.start, args.end, auto_adjust=args.auto_adjust)
    bench_ret = spy_close.pct_change().reindex(close.index).fillna(0.0)
    bench_eq = equity_curve(bench_ret)

    # Coerce everything to 1-D Series and align to close.index
    close = _to_series(close)
    signal = _to_series(signal, index=close.index).reindex(close.index)
    strat_ret = (
        _to_series(res["returns"], index=close.index).reindex(close.index).astype(float)
    )
    strat_eq = (
        _to_series(res["equity"], index=close.index).reindex(close.index).astype(float)
    )
    bench_ret = (
        _to_series(bench_ret, index=close.index)
        .reindex(close.index)
        .fillna(0.0)
        .astype(float)
    )
    bench_eq = (
        _to_series(bench_eq, index=close.index).reindex(close.index).astype(float)
    )

    # Align into one DataFrame for saving
    df = pd.DataFrame(
        {
            "close": close,
            "signal": signal,
            "strat_ret": strat_ret,
            "strat_eq": strat_eq,
            "bench_ret": bench_ret,
            "bench_eq": bench_eq,
        }
    )

    # Summaries
    strat_summary = summarize(
        df["strat_ret"], df["strat_eq"], df["signal"].shift(1).fillna(0.0)
    )
    bench_summary = {
        "N_days": int(len(bench_ret)),
        "AnnVol": float(bench_ret.std(ddof=1)) * (252**0.5),
        "Sharpe": sharpe(bench_ret),
        "Sortino": sortino(bench_ret),
        "CAGR": cagr(bench_ret),
        "MaxDD": max_drawdown(bench_eq),
    }

    # Print a compact report
    print(
        f"\nResults [{args.ticker}] {args.start} → {args.end}  (cost={args.cost_bps} bps)\n"
    )
    print(
        "Strategy:",
        f"Sharpe {strat_summary['Sharpe']:.2f} | Sortino {strat_summary['Sortino']:.2f} |",
        f"CAGR {strat_summary['CAGR']:.2%} | MaxDD {strat_summary['MaxDD']:.2%} |",
        f"Calmar {strat_summary['Calmar']:.2f} | TuW {strat_summary['TuW_days']}d |",
        f"Turnover {strat_summary['Turnover_annual']:.1f}/yr",
    )
    print(
        "SPY     :",
        f"Sharpe {bench_summary['Sharpe']:.2f} | Sortino {bench_summary['Sortino']:.2f} |",
        f"CAGR {bench_summary['CAGR']:.2%} | MaxDD {bench_summary['MaxDD']:.2%}\n",
    )

    # Save CSV of time series
    csv_path = outdir / f"{args.ticker}_results.csv"
    df.to_csv(csv_path, index=True)

    # Save JSON summary
    summary_path = outdir / f"{args.ticker}_summary.json"
    with summary_path.open("w") as f:
        json.dump(
            {"strategy": strat_summary, "benchmark_SPY": bench_summary}, f, indent=2
        )

    # Simple Markdown "results page"
    md = outdir / f"{args.ticker}_summary.md"
    md.write_text(
        f"""# Results — {args.ticker} ({args.start} → {args.end})

**Params:** SMA {args.fast}/{args.slow}, cost {args.cost_bps} bps, auto_adjust={args.auto_adjust}

## Strategy
- Sharpe: {strat_summary['Sharpe']:.2f}
- Sortino: {strat_summary['Sortino']:.2f}
- CAGR: {strat_summary['CAGR']:.2%}
- Max Drawdown: {strat_summary['MaxDD']:.2%}
- Calmar: {strat_summary['Calmar']:.2f}
- Time Under Water: {strat_summary['TuW_days']} days
- Turnover: {strat_summary['Turnover_annual']:.1f}/year

## Benchmark (SPY buy & hold)
- Sharpe: {bench_summary['Sharpe']:.2f}
- Sortino: {bench_summary['Sortino']:.2f}
- CAGR: {bench_summary['CAGR']:.2%}
- Max Drawdown: {bench_summary['MaxDD']:.2%}
"""
    )

    # Plots
    # 1) Price + SMAs
    fast_sma = close.rolling(args.fast, min_periods=args.fast).mean()
    slow_sma = close.rolling(args.slow, min_periods=args.slow).mean()
    plt.figure(figsize=(10, 4))
    close.plot(label=args.ticker)
    fast_sma.plot(label=f"SMA{args.fast}")
    slow_sma.plot(label=f"SMA{args.slow}")
    plt.title(f"{args.ticker} price & SMAs")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / f"{args.ticker}_price_sma.png")
    plt.close()

    # 2) Equity curves: strategy vs SPY
    plt.figure(figsize=(10, 4))
    df["strat_eq"].plot(label="Strategy")
    df["bench_eq"].plot(label="SPY")
    plt.title("Equity curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / f"{args.ticker}_equity.png")
    plt.close()

    # 3) Drawdown curve (strategy)
    peak = df["strat_eq"].cummax()
    dd = df["strat_eq"] / peak - 1.0
    plt.figure(figsize=(10, 2.8))
    dd.plot()
    plt.title("Strategy drawdown")
    plt.tight_layout()
    plt.savefig(outdir / f"{args.ticker}_drawdown.png")
    plt.close()


if __name__ == "__main__":
    main()
