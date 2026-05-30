import yfinance as yf
import pandas as pd


def _fetch_vix_intraday(period, interval, label):
    """
    Tries ^VIX first; falls back to VXX because Yahoo Finance often does not serve
    intraday data for index symbols. Returns a Close Series.
    """
    for ticker in ("^VIX", "VXX"):
        data = yf.download(ticker, period=period, interval=interval, progress=False)
        close = data["Close"] if not data.empty else pd.Series(dtype=float)
        if isinstance(close, pd.DataFrame):
            close = close.squeeze()
        if not close.empty:
            print(f"  VIX {label} source: {ticker}")
            return close
    raise RuntimeError(f"Could not fetch VIX intraday data ({label}) from ^VIX or VXX.")


def _fetch_vix_15m():
    return _fetch_vix_intraday(period="60d", interval="15m", label="15m")


def _fetch_4h_data(analysis="hybrid-spy"):
    """Fetches 1h data and resamples to 4h bars based on analysis mode."""
    need_spy = analysis in ("pure-spy", "hybrid-spy", "hybrid-vix")
    need_vix = analysis in ("pure-vix", "hybrid-spy", "hybrid-vix")
    df_parts = {}

    if need_spy:
        print("Fetching 1h SP500 futures data for 4h resample (last 180 days)...")
        sp500_1h = yf.download("ES=F", period="180d", interval="1h", progress=False)["Close"].squeeze()
        df_parts["SP500_Futures"] = sp500_1h.resample("4h").last().dropna()

    if need_vix:
        print("Fetching 1h VIX data for 4h resample (last 180 days)...")
        vix_1h = _fetch_vix_intraday(period="180d", interval="1h", label="4h(1h src)")
        df_parts["VIX_Futures"] = vix_1h.resample("4h").last().dropna()

    df_4h = pd.DataFrame(df_parts)
    df_4h.ffill(inplace=True)
    df_4h.dropna(inplace=True)
    price_col = "VIX_Futures" if analysis in ("pure-vix", "hybrid-vix") else "SP500_Futures"
    return add_technical_indicators(df_4h, price_col)


def fetch_and_prepare_data(timeframe="15m", analysis="hybrid-spy"):
    """Fetches market data based on analysis mode and returns (df_1d_tech, df_4h_tech, df_15m_tech).

    analysis modes:
      pure-spy   SP500 only  → predict SP500 direction
      pure-vix   VIX only    → predict VIX direction
      hybrid-spy SP500 + VIX → predict SP500 direction (default)
      hybrid-vix SP500 + VIX → predict VIX direction

    Unused tiers are returned as None:
      day → (df_1d, None,  None)
      4h  → (df_1d, df_4h, None)
      15m → (df_1d, df_4h, df_15m)
    """
    need_spy = analysis in ("pure-spy", "hybrid-spy", "hybrid-vix")
    need_vix = analysis in ("pure-vix", "hybrid-spy", "hybrid-vix")
    price_col = "VIX_Futures" if analysis in ("pure-vix", "hybrid-vix") else "SP500_Futures"

    # --- daily (always needed) ---
    daily_data = {}

    if need_spy:
        print("Fetching daily SP500 futures data (last 1 year)...")
        sp500_1d = yf.download("ES=F", period="1y", interval="1d", progress=False)["Close"].squeeze()
        sp500_1d.index = pd.to_datetime(sp500_1d.index.date)
        daily_data["SP500_Futures"] = sp500_1d

    if need_vix:
        print("Fetching daily VIX data (last 1 year)...")
        vix_1d = yf.download("^VIX", period="1y", interval="1d", progress=False)["Close"].squeeze()
        vix_1d.index = pd.to_datetime(vix_1d.index.date)
        daily_data["VIX_Futures"] = vix_1d

    df_1d = pd.DataFrame(daily_data)
    df_1d.ffill(inplace=True)
    df_1d.dropna(inplace=True)
    df_1d_tech = add_technical_indicators(df_1d, price_col)

    if timeframe == "day":
        print(f"Fetched {len(df_1d)} daily bars (day mode — intraday fetch skipped).")
        return df_1d_tech, None, None

    # --- 4h (needed for both '4h' and '15m' modes) ---
    df_4h_tech = _fetch_4h_data(analysis)

    if timeframe == "4h":
        print(f"Fetched {len(df_4h_tech)} 4h bars and {len(df_1d)} daily bars.")
        return df_1d_tech, df_4h_tech, None

    # --- 15-minute ---
    intraday_data = {}

    if need_spy:
        print("Fetching 15-minute SP500 futures data (last 60 days)...")
        sp500_15m = yf.download("ES=F", period="60d", interval="15m", progress=False)["Close"].squeeze()
        intraday_data["SP500_Futures"] = sp500_15m

    if need_vix:
        print("Fetching 15-minute VIX data (last 60 days)...")
        vix_15m = _fetch_vix_15m()
        intraday_data["VIX_Futures"] = vix_15m

    df_15m = pd.DataFrame(intraday_data)
    df_15m.ffill(inplace=True)
    df_15m.dropna(inplace=True)
    df_15m_tech = add_technical_indicators(df_15m, price_col)

    print(f"Fetched {len(df_15m)} 15m bars, {len(df_4h_tech)} 4h bars, and {len(df_1d)} daily bars.")
    return df_1d_tech, df_4h_tech, df_15m_tech


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
