import json
import pandas as pd

def serialize_for_ai(df_1d, df_15m):
    """
    Transforms raw DataFrames with technical indicators into a structured,
    token-efficient JSON string optimized for LLM prompt context injection.
    """
    # Ensure we are looking at the latest valid records
    latest_1d = df_1d.dropna().iloc[-1]
    latest_15m = df_15m.dropna().iloc[-1]
    
    # Compress the recent 15-minute history to give the AI context on momentum (last 5 ticks)
    recent_history = df_15m.dropna().tail(5)
    history_list = []
    for timestamp, row in recent_history.iterrows():
        history_list.append({
            "timestamp": str(timestamp),
            "sp500_close": round(row['SP500_Futures'], 2),
            "vix_close": round(row['VIX_Futures'], 2),
            "rsi": round(row['RSI'], 1),
            "macd_hist": round(row['MACD_Hist'], 2)
        })

    # Construct the unified, clean data structure
    ai_payload = {
        "metadata": {
            "asset_class": "Equity Index Futures / Volatility Futures",
            "tickers": ["ES=F", "VX=F"],
            "current_timestamp": str(df_15m.index[-1])
        },
        "macro_trend_1d": {
            "sp500_current": round(latest_1d['SP500_Futures'], 2),
            "vix_current": round(latest_1d['VIX_Futures'], 2),
            "moving_averages": {
                "ma20": round(latest_1d['MA_20'], 2),
                "ma60": round(latest_1d['MA_60'], 2),
                "ma90": round(latest_1d['MA_90'], 2),
                "structural_bias": "bullish" if latest_1d['SP500_Futures'] > latest_1d['MA_90'] else "bearish"
            },
            "rsi_1d": round(latest_1d['RSI'], 1)
        },
        "micro_execution_15m": {
            "sp500_current": round(latest_15m['SP500_Futures'], 2),
            "vix_current": round(latest_15m['VIX_Futures'], 2),
            "bollinger_bands": {
                "upper": round(latest_15m['BB_Upper'], 2),
                "lower": round(latest_15m['BB_Lower'], 2),
                "position_pct": round((latest_15m['SP500_Futures'] - latest_15m['BB_Lower']) / (latest_15m['BB_Upper'] - latest_15m['BB_Lower']) * 100, 1)
            },
            "momentum": {
                "rsi_15m": round(latest_15m['RSI'], 1),
                "macd_line": round(latest_15m['MACD_Line'], 2),
                "macd_signal": round(latest_15m['MACD_Signal'], 2),
                "macd_histogram": round(latest_15m['MACD_Hist'], 2)
            }
        },
        "recent_15m_ticks_history": history_list
    }
    
    # Return as a serialized JSON string
    return json.dumps(ai_payload, indent=2)

# Generate the payload
ai_ready_json = serialize_for_ai(df_1d_tech, df_15m_tech)
print(ai_ready_json)