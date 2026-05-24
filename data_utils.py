import yfinance as yf
import pandas as pd

# Define the Continuous Futures tickers
# ES=F: E-Mini S&P 500 Futures
# VX=F: VIX Futures
tickers = ["ES=F", "VX=F"]

# ==========================================
# 1. Fetch 15-minute tick data for 60 days
# ==========================================
print("Fetching 24-hour 15-minute futures data (last 60 days)...")
raw_15m = yf.download(tickers, period="60d", interval="15m")

# Extract Closing prices
df_15m = raw_15m['Close'][['ES=F', 'VX=F']].copy()
df_15m.columns = ['SP500_Futures', 'VIX_Futures']

# Forward-fill to handle minor synchronization gaps, then drop remaining NaNs
df_15m.ffill(inplace=True)
df_15m.dropna(inplace=True)

# ==========================================
# 2. Fetch daily tick data for 1 year
# ==========================================
print("Fetching daily futures data (last 1 year)...")
raw_1d = yf.download(tickers, period="1y", interval="1d")

# Extract Closing prices
df_1d = raw_1d['Close'][['ES=F', 'VX=F']].copy()
df_1d.columns = ['SP500_Futures', 'VIX_Futures']

# Clean up daily data
df_1d.dropna(inplace=True)

# ==========================================
# Review the Data Structures
# ==========================================
print("\n--- 24-Hour 15-Minute Data Structure ---")
print(f"Total rows: {len(df_15m)}")
print(df_15m.tail())

print("\n--- 24-Hour Daily Data Structure ---")
print(f"Total rows: {len(df_1d)}")
print(df_1d.tail())


def add_technical_indicators(df, column='SP500_Futures'):
    """
    Takes a DataFrame and calculates MA (20, 60, 90), Bollinger Bands, RSI, and MACD
    based on the specified price column.
    """
    # Create a copy to avoid SettingWithCopy warnings
    df_calc = df.copy()
    
    # ==========================================
    # 1. Simple Moving Averages (MA 20, 60, 90)
    # ==========================================
    df_calc['MA_20'] = df_calc[column].rolling(window=20).mean()
    df_calc['MA_60'] = df_calc[column].rolling(window=60).mean()
    df_calc['MA_90'] = df_calc[column].rolling(window=90).mean()
    
    # ==========================================
    # 2. 20-Period Bollinger Bands
    # ==========================================
    # Calculate the 20-period standard deviation
    std_20 = df_calc[column].rolling(window=20).std()
    
    # Upper and Lower bands are typically 2 standard deviations from the 20 MA
    df_calc['BB_Upper'] = df_calc['MA_20'] + (std_20 * 2)
    df_calc['BB_Lower'] = df_calc['MA_20'] - (std_20 * 2)
    
    # ==========================================
    # 3. Relative Strength Index (RSI)
    # ==========================================
    # Standard RSI uses a 14-period lookback. 
    delta = df_calc[column].diff()
    
    # Separate gains and losses
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # Calculate Wilder's smoothed moving average for gains and losses
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    df_calc['RSI'] = 100 - (100 / (1 + rs))
    
    # ==========================================
    # 4. MACD (Moving Average Convergence Divergence)
    # ==========================================
    # Standard MACD uses 12-period and 26-period Exponential Moving Averages (EMA)
    ema_12 = df_calc[column].ewm(span=12, adjust=False).mean()
    ema_26 = df_calc[column].ewm(span=26, adjust=False).mean()
    
    # MACD Line
    df_calc['MACD_Line'] = ema_12 - ema_26
    # Signal Line (9-period EMA of the MACD line)
    df_calc['MACD_Signal'] = df_calc['MACD_Line'].ewm(span=9, adjust=False).mean()
    # MACD Histogram (Difference between MACD and Signal)
    df_calc['MACD_Hist'] = df_calc['MACD_Line'] - df_calc['MACD_Signal']
    
    return df_calc

# Apply the function to your daily data
df_1d_tech = add_technical_indicators(df_1d, 'SP500_Futures')
vix_1da_tech = add_technical_indicators(df_1d, 'VIX_Futures')

# Apply the exact same function to your 15-minute data
df_15m_tech = add_technical_indicators(df_15m, 'SP500_Futures')
df_vix_15m_tech = add_technical_indicators(df_15m, 'VIX_Futures')

# Display the most recent rows of the daily data to verify the new columns
print("Daily Data with Technical Indicators:")
# Dropping NaN values temporarily just for the print output (since early rows will lack 90-day MA data)
print(df_1d_tech.dropna().tail())