import yfinance as yf
import pandas as pd


def _fetch_vix_15m():
    """
    Tries ^VIX first for intraday VIX; falls back to VXX (VIX short-term futures ETN)
    because Yahoo Finance often does not serve intraday data for index symbols.
    """
    for ticker in ("^VIX", "VXX"):
        data = yf.download(ticker, period="60d", interval="15m", progress=False)
        close = data["Close"] if not data.empty else pd.Series(dtype=float)
        # squeeze away the ticker-level column if yfinance returns a DataFrame
        if isinstance(close, pd.DataFrame):
            close = close.squeeze()
        if not close.empty:
            print(f"  VIX 15m source: {ticker}")
            return close
    raise RuntimeError("Could not fetch VIX intraday data from ^VIX or VXX.")


def fetch_and_prepare_data():
    """Fetches SP500 (ES=F) and VIX (^VIX) data and returns (df_1d_tech, df_15m_tech) with indicators."""

    # --- 15-minute ---
    print("Fetching 15-minute SP500 futures data (last 60 days)...")
    sp500_15m = yf.download("ES=F", period="60d", interval="15m", progress=False)["Close"].squeeze()

    print("Fetching 15-minute VIX data (last 60 days)...")
    vix_15m = _fetch_vix_15m()

    df_15m = pd.DataFrame({"SP500_Futures": sp500_15m, "VIX_Futures": vix_15m})
    df_15m.ffill(inplace=True)
    df_15m.dropna(inplace=True)

    # --- daily ---
    print("Fetching daily SP500 futures data (last 1 year)...")
    sp500_1d = yf.download("ES=F", period="1y", interval="1d", progress=False)["Close"].squeeze()

    print("Fetching daily VIX data (last 1 year)...")
    vix_1d = yf.download("^VIX", period="1y", interval="1d", progress=False)["Close"].squeeze()

    df_1d = pd.DataFrame({"SP500_Futures": sp500_1d, "VIX_Futures": vix_1d})
    df_1d.dropna(inplace=True)

    print(f"Fetched {len(df_15m)} 15m bars and {len(df_1d)} daily bars.")

    df_1d_tech = add_technical_indicators(df_1d, "SP500_Futures")
    df_15m_tech = add_technical_indicators(df_15m, "SP500_Futures")

    return df_1d_tech, df_15m_tech


def add_technical_indicators(df, column="SP500_Futures"):
    """Calculates MA (20, 60, 90), Bollinger Bands, RSI, and MACD on the given price column."""
    df_calc = df.copy()

    df_calc["MA_20"] = df_calc[column].rolling(window=20).mean()
    df_calc["MA_60"] = df_calc[column].rolling(window=60).mean()
    df_calc["MA_90"] = df_calc[column].rolling(window=90).mean()

    std_20 = df_calc[column].rolling(window=20).std()
    df_calc["BB_Upper"] = df_calc["MA_20"] + (std_20 * 2)
    df_calc["BB_Lower"] = df_calc["MA_20"] - (std_20 * 2)

    delta = df_calc[column].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False).mean()
    rs = avg_gain / avg_loss
    df_calc["RSI"] = 100 - (100 / (1 + rs))

    ema_12 = df_calc[column].ewm(span=12, adjust=False).mean()
    ema_26 = df_calc[column].ewm(span=26, adjust=False).mean()
    df_calc["MACD_Line"] = ema_12 - ema_26
    df_calc["MACD_Signal"] = df_calc["MACD_Line"].ewm(span=9, adjust=False).mean()
    df_calc["MACD_Hist"] = df_calc["MACD_Line"] - df_calc["MACD_Signal"]

    return df_calc
