import json
import time

from ai_agent_connector import get_ai_market_signal


def serialize_for_ai(df_1d, df_15m):
    """
    Transforms DataFrames with technical indicators into a structured,
    token-efficient JSON string for LLM prompt injection.
    """
    latest_1d = df_1d.dropna().iloc[-1]
    latest_15m = df_15m.dropna().iloc[-1]

    recent_history = df_15m.dropna().tail(5)
    history_list = []
    for timestamp, row in recent_history.iterrows():
        history_list.append({
            "timestamp": str(timestamp),
            "sp500_close": round(row["SP500_Futures"], 2),
            "vix_close": round(row["VIX_Futures"], 2),
            "rsi": round(row["RSI"], 1),
            "macd_hist": round(row["MACD_Hist"], 2),
        })

    ai_payload = {
        "metadata": {
            "asset_class": "Equity Index Futures / Volatility Futures",
            "tickers": ["ES=F", "VX=F"],
            "current_timestamp": str(df_15m.index[-1]),
        },
        "macro_trend_1d": {
            "sp500_current": round(latest_1d["SP500_Futures"], 2),
            "vix_current": round(latest_1d["VIX_Futures"], 2),
            "moving_averages": {
                "ma20": round(latest_1d["MA_20"], 2),
                "ma60": round(latest_1d["MA_60"], 2),
                "ma90": round(latest_1d["MA_90"], 2),
                "structural_bias": (
                    "bullish" if latest_1d["SP500_Futures"] > latest_1d["MA_90"] else "bearish"
                ),
            },
            "rsi_1d": round(latest_1d["RSI"], 1),
        },
        "micro_execution_15m": {
            "sp500_current": round(latest_15m["SP500_Futures"], 2),
            "vix_current": round(latest_15m["VIX_Futures"], 2),
            "bollinger_bands": {
                "upper": round(latest_15m["BB_Upper"], 2),
                "lower": round(latest_15m["BB_Lower"], 2),
                "position_pct": round(
                    (latest_15m["SP500_Futures"] - latest_15m["BB_Lower"])
                    / (latest_15m["BB_Upper"] - latest_15m["BB_Lower"])
                    * 100,
                    1,
                ),
            },
            "momentum": {
                "rsi_15m": round(latest_15m["RSI"], 1),
                "macd_line": round(latest_15m["MACD_Line"], 2),
                "macd_signal": round(latest_15m["MACD_Signal"], 2),
                "macd_histogram": round(latest_15m["MACD_Hist"], 2),
            },
        },
        "recent_15m_ticks_history": history_list,
    }

    return json.dumps(ai_payload, indent=2)


def run_historical_orchestration(
    df_1d_tech, df_15m_tech, backtester, test_window=20, provider="openai", model_name=None
):
    """
    Simulates live execution by walking forward through historical data,
    calling the AI for each bar and logging the result in the backtester.
    """
    print(f"\n{'=' * 55}")
    print(f"  STARTING AI BACKTEST LOOP  |  window={test_window} bars")
    print(f"{'=' * 55}")

    df_15m_clean = df_15m_tech.dropna()
    total_rows = len(df_15m_clean)

    start_idx = total_rows - test_window - backtester.lookahead
    end_idx = total_rows - backtester.lookahead

    if start_idx < 0:
        print("Error: test_window + lookahead exceeds available data rows.")
        return

    for i in range(start_idx, end_idx):
        current_15m_slice = df_15m_clean.iloc[: i + 1]
        current_timestamp = current_15m_slice.index[-1]

        ai_payload = serialize_for_ai(df_1d_tech, current_15m_slice)

        print(f"[{current_timestamp}] Querying {provider.upper()}...")
        try:
            raw_response = get_ai_market_signal(ai_payload, provider=provider, model_name=model_name)

            if "BUY" in raw_response.upper():
                signal = "BUY"
            elif "SELL" in raw_response.upper():
                signal = "SELL"
            else:
                signal = "HOLD"

        except Exception as e:
            print(f"  API error: {e} — defaulting to HOLD")
            signal = "HOLD"

        print(f"  Signal: {signal}")

        trade_result = backtester.evaluate_signal(current_timestamp, signal)

        if trade_result and signal != "HOLD":
            status = "WON" if trade_result["is_correct"] else "LOST"
            print(f"  Result: {status}  ({trade_result['trade_return_pct']}% return)")

        time.sleep(1)

    print(f"\n{'=' * 55}")
    print("  BACKTEST COMPLETE — FINAL METRICS")
    print(f"{'=' * 55}")

    metrics = backtester.calculate_performance_metrics()
    print(json.dumps(metrics, indent=4))
