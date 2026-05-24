import json
import time

from ai_agent_connector import get_ai_market_signal


def serialize_for_ai_day(df_1d):
    """Serializes daily-only data for the AI — used in day mode to predict next trading day's direction."""
    latest = df_1d.dropna().iloc[-1]

    recent_history = df_1d.dropna().tail(5)
    history_list = []
    for timestamp, row in recent_history.iterrows():
        history_list.append({
            "date": str(timestamp)[:10],
            "sp500_close": round(row["SP500_Futures"], 2),
            "vix_close": round(row["VIX_Futures"], 2),
            "rsi": round(row["RSI"], 1),
            "macd_hist": round(row["MACD_Hist"], 2),
        })

    bb_range = latest["BB_Upper"] - latest["BB_Lower"]
    ai_payload = {
        "metadata": {
            "asset_class": "Equity Index Futures / Volatility Index",
            "tickers": ["ES=F", "^VIX"],
            "current_date": str(df_1d.index[-1])[:10],
            "prediction_target": "next_trading_day_direction",
        },
        "daily_trend": {
            "sp500_current": round(latest["SP500_Futures"], 2),
            "vix_current": round(latest["VIX_Futures"], 2),
            "moving_averages": {
                "ma20": round(latest["MA_20"], 2),
                "ma60": round(latest["MA_60"], 2),
                "ma90": round(latest["MA_90"], 2),
                "structural_bias": (
                    "bullish" if latest["SP500_Futures"] > latest["MA_90"] else "bearish"
                ),
            },
            "bollinger_bands": {
                "upper": round(latest["BB_Upper"], 2),
                "lower": round(latest["BB_Lower"], 2),
                "position_pct": round(
                    (latest["SP500_Futures"] - latest["BB_Lower"]) / bb_range * 100, 1
                ) if bb_range != 0 else 50.0,
            },
            "momentum": {
                "rsi_1d": round(latest["RSI"], 1),
                "macd_line": round(latest["MACD_Line"], 2),
                "macd_signal": round(latest["MACD_Signal"], 2),
                "macd_histogram": round(latest["MACD_Hist"], 2),
            },
        },
        "recent_daily_history": history_list,
    }

    return json.dumps(ai_payload, indent=2)


def serialize_for_ai(df_1d, df_4h, df_15m):
    """
    3-tier MTA payload: Macro (1d) → Swing (4h) → Micro (15m).
    """
    latest_1d = df_1d.dropna().iloc[-1]
    latest_4h = df_4h.dropna().iloc[-1]
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

    bb_range_4h = latest_4h["BB_Upper"] - latest_4h["BB_Lower"]
    bb_range_15m = latest_15m["BB_Upper"] - latest_15m["BB_Lower"]

    ai_payload = {
        "metadata": {
            "asset_class": "Equity Index Futures / Volatility Futures",
            "tickers": ["ES=F", "^VIX"],
            "current_timestamp": str(df_15m.index[-1]),
            "prediction_target": "next_15min_bar_direction",
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
        "swing_context_4h": {
            "sp500_current": round(latest_4h["SP500_Futures"], 2),
            "vix_current": round(latest_4h["VIX_Futures"], 2),
            "bollinger_bands": {
                "upper": round(latest_4h["BB_Upper"], 2),
                "lower": round(latest_4h["BB_Lower"], 2),
                "position_pct": round(
                    (latest_4h["SP500_Futures"] - latest_4h["BB_Lower"]) / bb_range_4h * 100, 1
                ) if bb_range_4h != 0 else 50.0,
            },
            "momentum": {
                "rsi_4h": round(latest_4h["RSI"], 1),
                "macd_line": round(latest_4h["MACD_Line"], 2),
                "macd_signal": round(latest_4h["MACD_Signal"], 2),
                "macd_histogram": round(latest_4h["MACD_Hist"], 2),
            },
        },
        "micro_execution_15m": {
            "sp500_current": round(latest_15m["SP500_Futures"], 2),
            "vix_current": round(latest_15m["VIX_Futures"], 2),
            "bollinger_bands": {
                "upper": round(latest_15m["BB_Upper"], 2),
                "lower": round(latest_15m["BB_Lower"], 2),
                "position_pct": round(
                    (latest_15m["SP500_Futures"] - latest_15m["BB_Lower"]) / bb_range_15m * 100, 1
                ) if bb_range_15m != 0 else 50.0,
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


def serialize_for_ai_4h(df_1d, df_4h):
    """Serializes daily + 4h data for the AI — predicts next 4-hour bar's direction."""
    latest_1d = df_1d.dropna().iloc[-1]
    latest_4h = df_4h.dropna().iloc[-1]

    recent_history = df_4h.dropna().tail(5)
    history_list = []
    for timestamp, row in recent_history.iterrows():
        history_list.append({
            "timestamp": str(timestamp),
            "sp500_close": round(row["SP500_Futures"], 2),
            "vix_close": round(row["VIX_Futures"], 2),
            "rsi": round(row["RSI"], 1),
            "macd_hist": round(row["MACD_Hist"], 2),
        })

    bb_range = latest_4h["BB_Upper"] - latest_4h["BB_Lower"]
    ai_payload = {
        "metadata": {
            "asset_class": "Equity Index Futures / Volatility Futures",
            "tickers": ["ES=F", "^VIX"],
            "current_timestamp": str(df_4h.index[-1]),
            "prediction_target": "next_4h_bar_direction",
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
        "swing_execution_4h": {
            "sp500_current": round(latest_4h["SP500_Futures"], 2),
            "vix_current": round(latest_4h["VIX_Futures"], 2),
            "bollinger_bands": {
                "upper": round(latest_4h["BB_Upper"], 2),
                "lower": round(latest_4h["BB_Lower"], 2),
                "position_pct": round(
                    (latest_4h["SP500_Futures"] - latest_4h["BB_Lower"]) / bb_range * 100, 1
                ) if bb_range != 0 else 50.0,
            },
            "momentum": {
                "rsi_4h": round(latest_4h["RSI"], 1),
                "macd_line": round(latest_4h["MACD_Line"], 2),
                "macd_signal": round(latest_4h["MACD_Signal"], 2),
                "macd_histogram": round(latest_4h["MACD_Hist"], 2),
            },
        },
        "recent_4h_ticks_history": history_list,
    }

    return json.dumps(ai_payload, indent=2)


def run_historical_orchestration(
    df_1d_tech, df_15m_tech, backtester, test_window=20, provider="openai", model_name=None, timeframe="15m", df_4h_tech=None
):
    """
    Simulates live execution by walking forward through historical data,
    calling the AI for each bar and logging the result in the backtester.
    """
    print(f"\n{'=' * 55}")
    mode_labels = {"day": "Day", "4h": "4-Hour", "15m": "15-Min"}
    mode_label = mode_labels.get(timeframe, timeframe)
    print(f"  STARTING AI BACKTEST LOOP  |  mode={mode_label}  |  window={test_window} bars")
    print(f"{'=' * 55}")

    if timeframe == "day":
        df_walk = df_1d_tech.dropna()
    elif timeframe == "4h":
        df_walk = df_4h_tech.dropna()
    else:
        df_walk = df_15m_tech.dropna()
    total_rows = len(df_walk)

    start_idx = total_rows - test_window - backtester.lookahead
    end_idx = total_rows - backtester.lookahead

    if start_idx < 0:
        print("Error: test_window + lookahead exceeds available data rows.")
        return

    for i in range(start_idx, end_idx):
        current_slice = df_walk.iloc[: i + 1]
        current_timestamp = current_slice.index[-1]

        if timeframe == "day":
            ai_payload = serialize_for_ai_day(current_slice)
        elif timeframe == "4h":
            ai_payload = serialize_for_ai_4h(df_1d_tech, current_slice)
        else:
            ai_payload = serialize_for_ai(df_1d_tech, df_4h_tech, current_slice)

        print(f"[{current_timestamp}] Querying {provider.upper()}...")
        try:
            raw_response = get_ai_market_signal(
                ai_payload, provider=provider, model_name=model_name, timeframe=timeframe
            )

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
            future_date = str(trade_result['future_timestamp'])[:10]
            print(f"  Entry: {trade_result['entry_price']}  →  Future [{future_date}]: {trade_result['future_price']}  (Δ {trade_result['price_change_pct']}%)")
            print(f"  Result: {status}  ({trade_result['trade_return_pct']}% return)")

        time.sleep(1)

    print(f"\n{'=' * 55}")
    print("  BACKTEST COMPLETE — FINAL METRICS")
    print(f"{'=' * 55}")

    metrics = backtester.calculate_performance_metrics()
    print(json.dumps(metrics, indent=4))
