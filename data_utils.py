import yfinance as yf
import pandas as pd


def fetch_and_prepare_data():
    """Fetches raw futures data from yfinance and returns (df_1d_tech, df_15m_tech) with indicators applied."""
    tickers = ["ES=F", "VX=F"]

    print("Fetching 15-minute futures data (last 60 days)...")
    raw_15m = yf.download(tickers, period="60d", interval="15m", progress=False)
    df_15m = raw_15m["Close"][["ES=F", "VX=F"]].copy()
    df_15m.columns = ["SP500_Futures", "VIX_Futures"]
    df_15m.ffill(inplace=True)
    df_15m.dropna(inplace=True)

    print("Fetching daily futures data (last 1 year)...")
    raw_1d = yf.download(tickers, period="1y", interval="1d", progress=False)
    df_1d = raw_1d["Close"][["ES=F", "VX=F"]].copy()
    df_1d.columns = ["SP500_Futures", "VIX_Futures"]
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
