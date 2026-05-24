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

import time
import json
import pandas as pd

def run_historical_orchestration(df_1d_tech, df_15m_tech, backtester, test_window=20):
    """
    Simulates live execution by walking forward through the historical data.
    
    Args:
        df_1d_tech (DataFrame): The daily macro dataframe.
        df_15m_tech (DataFrame): The 15m tactical dataframe.
        backtester (MarketAIBacktester): The initialized backtesting engine.
        test_window (int): How many recent 15m bars to test (e.g., 20 bars = 5 hours).
    """
    print(f"\n{'='*50}")
    print(f"🚀 STARTING AI ORCHESTRATION LOOP ({test_window} PERIODS)")
    print(f"{'='*50}")
    
    # Ensure data is clean
    df_15m_clean = df_15m_tech.dropna()
    total_rows = len(df_15m_clean)
    
    # We need to stop the loop before the very end of the dataframe, 
    # otherwise the backtester won't have future data to verify the final trades.
    start_idx = total_rows - test_window - backtester.lookahead
    end_idx = total_rows - backtester.lookahead
    
    if start_idx < 0:
        print("Error: test_window and lookahead are larger than available data.")
        return

    # Walk forward through time
    for i in range(start_idx, end_idx):
        # 1. Slice the data to simulate "current" time (blinding the AI to the future)
        current_15m_slice = df_15m_clean.iloc[:i+1]
        current_timestamp = current_15m_slice.index[-1]
        
        # 2. Generate the API Payload
        # (We use the full 1d dataframe here assuming the macro trend doesn't change intraday,
        # but in strict production, you'd slice the 1d dataframe to match the date as well.)
        ai_payload = serialize_for_ai(df_1d_tech, current_15m_slice)
        
        # 3. Call the AI API
        print(f"[{current_timestamp}] Analyzing market state...")
        try:
            # We use Qwen, DeepSeek, or OpenAI based on your configuration from Step 3
            # NOTE: For large backtests, keep API rate limits in mind!
            raw_ai_response = get_ai_market_signal(ai_payload, provider="openai")
            
            # 4. Parse the Signal 
            # We extract the exact word (BUY, SELL, HOLD) from the AI's structured text
            if "BUY" in raw_ai_response.upper():
                signal = "BUY"
            elif "SELL" in raw_ai_response.upper():
                signal = "SELL"
            else:
                signal = "HOLD"
                
        except Exception as e:
            print(f"API Error at {current_timestamp}: {e}")
            signal = "HOLD" # Fail safe
            
        print(f"↳ Signal Generated: {signal}")
        
        # 5. Log it into the Backtester
        trade_result = backtester.evaluate_signal(current_timestamp, signal)
        
        if trade_result and signal != "HOLD":
            status = "✅ WON" if trade_result['is_correct'] else "❌ LOST"
            print(f"  ↳ Result: {status} ({trade_result['trade_return_pct']}% return)")
            
        # Optional: Sleep briefly to avoid tripping free-tier API rate limits
        time.sleep(1) 

    # 6. Final Evaluation
    print(f"\n{'='*50}")
    print("📊 BACKTEST COMPLETE - FINAL METRICS")
    print(f"{'='*50}")
    
    metrics = backtester.calculate_performance_metrics()
    print(json.dumps(metrics, indent=4))


# =====================================================================
# Execution Block
# =====================================================================
if __name__ == "__main__":
    # 1. Initialize the backtester looking forward 16 periods (4 hours)
    live_backtester = MarketAIBacktester(df_15m_tech, lookahead_periods=16)
    
    # 2. Run the loop for the last 20 periods 
    # (Testing ~5 hours of market data. Increase test_window for a deeper backtest).
    run_historical_orchestration(
        df_1d_tech=df_1d_tech, 
        df_15m_tech=df_15m_tech, 
        backtester=live_backtester, 
        test_window=20
    )