import argparse
import sys
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")


def _extract_eps(stmt, candidates=("Diluted EPS", "Basic EPS")) -> pd.Series | None:
    for name in candidates:
        if name in stmt.index:
            return stmt.loc[name].dropna().sort_index()
    return None


def fetch_pe_history(ticker_symbol: str, period_years: int = 10) -> pd.Series:
    """
    Returns a daily P/E series. TTM EPS is built from two sources:
      1. Annual income statement  → fiscal-year EPS = TTM EPS at year-end (~4 yrs)
      2. Quarterly income statement → rolling 4-quarter sum (~1-2 recent quarters)
    The two are merged (quarterly preferred where dates overlap) then forward-filled daily.
    """
    ticker = yf.Ticker(ticker_symbol)
    ttm_parts: list[pd.Series] = []

    # Annual EPS: each year-end value is itself a TTM EPS
    try:
        annual_eps = _extract_eps(ticker.income_stmt)
        if annual_eps is not None and not annual_eps.empty:
            ttm_parts.append(annual_eps)
    except Exception:
        pass

    # Quarterly TTM: rolling 4-quarter sum
    try:
        q_eps = _extract_eps(ticker.quarterly_income_stmt)
        if q_eps is not None and len(q_eps) >= 4:
            ttm_parts.append(q_eps.rolling(4).sum().dropna())
    except Exception:
        pass

    if not ttm_parts:
        sys.exit("Could not retrieve EPS data from yfinance for this ticker.")

    # Merge: concat then de-duplicate (last value wins → quarterly beats annual on same date)
    ttm_eps = pd.concat(ttm_parts).sort_index()
    ttm_eps = ttm_eps[~ttm_eps.index.duplicated(keep="last")]

    print(f"  TTM EPS range: {ttm_eps.index[0].date()} → {ttm_eps.index[-1].date()} ({len(ttm_eps)} data points)")

    # --- daily price history ---
    end = datetime.today()
    start = pd.Timestamp(end) - pd.DateOffset(years=period_years)
    hist = ticker.history(start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"), auto_adjust=True)
    if hist.empty:
        sys.exit(f"No price history found for {ticker_symbol}.")

    price = hist["Close"]
    # Normalize to tz-naive so both indexes share the same dtype for reindexing
    price.index = price.index.tz_localize(None)

    # Reindex TTM EPS onto the daily price index.
    # Each quarter's TTM value is held forward (ffill) until the next report.
    ttm_daily = ttm_eps.reindex(price.index, method="ffill")

    # Drop days before first TTM estimate or where EPS <= 0 (avoid nonsensical ratios)
    valid = ttm_daily.notna() & (ttm_daily > 0)
    pe = (price[valid] / ttm_daily[valid]).rename("P/E")
    return pe


def plot_pe(ticker_symbol: str, pe: pd.Series, save_path: str | None = None) -> None:
    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(pe.index, pe.values, linewidth=1.5, color="#2196F3", label="Trailing P/E")

    # Shaded percentile bands for context
    p25, median, p75 = pe.quantile([0.25, 0.50, 0.75])
    ax.axhline(median, color="#FF9800", linewidth=1, linestyle="--", label=f"Median {median:.1f}x")
    ax.fill_between(pe.index, p25, p75, alpha=0.12, color="#FF9800", label=f"25th–75th pct ({p25:.1f}–{p75:.1f}x)")

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    fig.autofmt_xdate()

    ax.set_title(f"{ticker_symbol.upper()} — Trailing Twelve-Month P/E Ratio", fontsize=14, fontweight="bold")
    ax.set_ylabel("P/E Ratio (x)")
    ax.set_xlabel("Date")
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Chart saved to {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot historical P/E ratio for a stock ticker.")
    parser.add_argument("ticker", help="Stock ticker symbol, e.g. AAPL")
    parser.add_argument("--years", type=int, default=10, help="Years of history to plot (default: 10)")
    parser.add_argument("--save", metavar="FILE", help="Save chart to file instead of displaying it (e.g. aapl_pe.png)")
    args = parser.parse_args()

    print(f"Fetching P/E data for {args.ticker.upper()} …")
    pe = fetch_pe_history(args.ticker.upper(), args.years)

    if pe.empty:
        sys.exit("No valid P/E data after filtering.")

    print(f"  Date range : {pe.index[0].date()} → {pe.index[-1].date()}")
    print(f"  Current P/E: {pe.iloc[-1]:.1f}x")
    print(f"  Median P/E : {pe.median():.1f}x")
    print(f"  Min / Max  : {pe.min():.1f}x / {pe.max():.1f}x")

    plot_pe(args.ticker.upper(), pe, save_path=args.save)


if __name__ == "__main__":
    main()
