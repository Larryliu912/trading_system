import json
import re
import time

from ai_agent_connector import get_ai_market_signal


def _history_entry_day(timestamp, row, include_spy, include_vix):
    entry = {"date": str(timestamp)[:10]}
    if include_spy:
        entry["sp500_close"] = round(row["SP500_Futures"], 2)
    if include_vix:
        entry["vix_close"] = round(row["VIX_Futures"], 2)
    entry["rsi"] = round(row["RSI"], 1)
    entry["macd_hist"] = round(row["MACD_Hist"], 2)
    return entry


def _history_entry_intraday(timestamp, row, include_spy, include_vix):
    entry = {"timestamp": str(timestamp)}
    if include_spy:
        entry["sp500_close"] = round(row["SP500_Futures"], 2)
    if include_vix:
        entry["vix_close"] = round(row["VIX_Futures"], 2)
    entry["rsi"] = round(row["RSI"], 1)
    entry["macd_hist"] = round(row["MACD_Hist"], 2)
    return entry


def _predict_target_label(analysis, horizon):
    asset = "VIX" if analysis in ("pure-vix", "hybrid-vix") else "SP500"
    return f"next_{horizon}_{asset}_direction"


def serialize_for_ai_day(df_1d, analysis="hybrid-spy"):
    """Serializes daily data for the AI — predicts next trading day's direction."""
    df = df_1d.dropna()
    latest = df.iloc[-1]

    include_spy = analysis in ("pure-spy", "hybrid-spy", "hybrid-vix")
    include_vix = analysis in ("pure-vix", "hybrid-spy", "hybrid-vix")
    price_col = "VIX_Futures" if analysis in ("pure-vix", "hybrid-vix") else "SP500_Futures"

    history_list = [
        _history_entry_day(ts, row, include_spy, include_vix)
        for ts, row in df.tail(5).iterrows()
    ]

    bb_range = latest["BB_Upper"] - latest["BB_Lower"]
    daily_trend = {
        "moving_averages": {
            "ma20": round(latest["MA_20"], 2),
            "ma60": round(latest["MA_60"], 2),
            "ma90": round(latest["MA_90"], 2),
            "structural_bias": "bullish" if latest[price_col] > latest["MA_90"] else "bearish",
        },
        "bollinger_bands": {
            "upper": round(latest["BB_Upper"], 2),
            "lower": round(latest["BB_Lower"], 2),
            "position_pct": round(
                (latest[price_col] - latest["BB_Lower"]) / bb_range * 100, 1
            ) if bb_range != 0 else 50.0,
        },
        "momentum": {
            "rsi_1d": round(latest["RSI"], 1),
            "macd_line": round(latest["MACD_Line"], 2),
            "macd_signal": round(latest["MACD_Signal"], 2),
            "macd_histogram": round(latest["MACD_Hist"], 2),
        },
    }
    if include_spy:
        daily_trend["sp500_current"] = round(latest["SP500_Futures"], 2)
    if include_vix:
        daily_trend["vix_current"] = round(latest["VIX_Futures"], 2)

    ai_payload = {
        "metadata": {
            "analysis_mode": analysis,
            "current_date": str(df.index[-1])[:10],
            "prediction_target": _predict_target_label(analysis, "trading_day"),
        },
        "daily_trend": daily_trend,
        "recent_daily_history": history_list,
    }
    return json.dumps(ai_payload, indent=2)


def serialize_for_ai_4h(df_1d, df_4h, analysis="hybrid-spy"):
    """Serializes daily + 4h data for the AI — predicts next 4-hour bar's direction."""
    latest_1d = df_1d.dropna().iloc[-1]
    latest_4h = df_4h.dropna().iloc[-1]

    include_spy = analysis in ("pure-spy", "hybrid-spy", "hybrid-vix")
    include_vix = analysis in ("pure-vix", "hybrid-spy", "hybrid-vix")
    price_col = "VIX_Futures" if analysis in ("pure-vix", "hybrid-vix") else "SP500_Futures"

    history_list = [
        _history_entry_intraday(ts, row, include_spy, include_vix)
        for ts, row in df_4h.dropna().tail(5).iterrows()
    ]

    bb_range = latest_4h["BB_Upper"] - latest_4h["BB_Lower"]

    macro_trend = {
        "moving_averages": {
            "ma20": round(latest_1d["MA_20"], 2),
            "ma60": round(latest_1d["MA_60"], 2),
            "ma90": round(latest_1d["MA_90"], 2),
            "structural_bias": "bullish" if latest_1d[price_col] > latest_1d["MA_90"] else "bearish",
        },
        "rsi_1d": round(latest_1d["RSI"], 1),
    }
    if include_spy:
        macro_trend["sp500_current"] = round(latest_1d["SP500_Futures"], 2)
    if include_vix:
        macro_trend["vix_current"] = round(latest_1d["VIX_Futures"], 2)

    swing_exec = {
        "bollinger_bands": {
            "upper": round(latest_4h["BB_Upper"], 2),
            "lower": round(latest_4h["BB_Lower"], 2),
            "position_pct": round(
                (latest_4h[price_col] - latest_4h["BB_Lower"]) / bb_range * 100, 1
            ) if bb_range != 0 else 50.0,
        },
        "momentum": {
            "rsi_4h": round(latest_4h["RSI"], 1),
            "macd_line": round(latest_4h["MACD_Line"], 2),
            "macd_signal": round(latest_4h["MACD_Signal"], 2),
            "macd_histogram": round(latest_4h["MACD_Hist"], 2),
        },
    }
    if include_spy:
        swing_exec["sp500_current"] = round(latest_4h["SP500_Futures"], 2)
    if include_vix:
        swing_exec["vix_current"] = round(latest_4h["VIX_Futures"], 2)

    ai_payload = {
        "metadata": {
            "analysis_mode": analysis,
            "current_timestamp": str(df_4h.index[-1]),
            "prediction_target": _predict_target_label(analysis, "4h_bar"),
        },
        "macro_trend_1d": macro_trend,
        "swing_execution_4h": swing_exec,
        "recent_4h_ticks_history": history_list,
    }
    return json.dumps(ai_payload, indent=2)


def serialize_for_ai(df_1d, df_4h, df_15m, analysis="hybrid-spy"):
    """Serializes 3-tier MTA payload (1d + 4h + 15m) — predicts next 15-min bar's direction."""
    latest_1d = df_1d.dropna().iloc[-1]
    latest_4h = df_4h.dropna().iloc[-1]
    latest_15m = df_15m.dropna().iloc[-1]

    include_spy = analysis in ("pure-spy", "hybrid-spy", "hybrid-vix")
    include_vix = analysis in ("pure-vix", "hybrid-spy", "hybrid-vix")
    price_col = "VIX_Futures" if analysis in ("pure-vix", "hybrid-vix") else "SP500_Futures"

    history_list = [
        _history_entry_intraday(ts, row, include_spy, include_vix)
        for ts, row in df_15m.dropna().tail(5).iterrows()
    ]

    bb_range_4h = latest_4h["BB_Upper"] - latest_4h["BB_Lower"]
    bb_range_15m = latest_15m["BB_Upper"] - latest_15m["BB_Lower"]

    macro_trend = {
        "moving_averages": {
            "ma20": round(latest_1d["MA_20"], 2),
            "ma60": round(latest_1d["MA_60"], 2),
            "ma90": round(latest_1d["MA_90"], 2),
            "structural_bias": "bullish" if latest_1d[price_col] > latest_1d["MA_90"] else "bearish",
        },
        "rsi_1d": round(latest_1d["RSI"], 1),
    }
    if include_spy:
        macro_trend["sp500_current"] = round(latest_1d["SP500_Futures"], 2)
    if include_vix:
        macro_trend["vix_current"] = round(latest_1d["VIX_Futures"], 2)

    swing_context = {
        "bollinger_bands": {
            "upper": round(latest_4h["BB_Upper"], 2),
            "lower": round(latest_4h["BB_Lower"], 2),
            "position_pct": round(
                (latest_4h[price_col] - latest_4h["BB_Lower"]) / bb_range_4h * 100, 1
            ) if bb_range_4h != 0 else 50.0,
        },
        "momentum": {
            "rsi_4h": round(latest_4h["RSI"], 1),
            "macd_line": round(latest_4h["MACD_Line"], 2),
            "macd_signal": round(latest_4h["MACD_Signal"], 2),
            "macd_histogram": round(latest_4h["MACD_Hist"], 2),
        },
    }
    if include_spy:
        swing_context["sp500_current"] = round(latest_4h["SP500_Futures"], 2)
    if include_vix:
        swing_context["vix_current"] = round(latest_4h["VIX_Futures"], 2)

    micro_exec = {
        "bollinger_bands": {
            "upper": round(latest_15m["BB_Upper"], 2),
            "lower": round(latest_15m["BB_Lower"], 2),
            "position_pct": round(
                (latest_15m[price_col] - latest_15m["BB_Lower"]) / bb_range_15m * 100, 1
            ) if bb_range_15m != 0 else 50.0,
        },
        "momentum": {
            "rsi_15m": round(latest_15m["RSI"], 1),
            "macd_line": round(latest_15m["MACD_Line"], 2),
            "macd_signal": round(latest_15m["MACD_Signal"], 2),
            "macd_histogram": round(latest_15m["MACD_Hist"], 2),
        },
    }
    if include_spy:
        micro_exec["sp500_current"] = round(latest_15m["SP500_Futures"], 2)
    if include_vix:
        micro_exec["vix_current"] = round(latest_15m["VIX_Futures"], 2)

    ai_payload = {
        "metadata": {
            "analysis_mode": analysis,
            "current_timestamp": str(df_15m.index[-1]),
            "prediction_target": _predict_target_label(analysis, "15min_bar"),
        },
        "macro_trend_1d": macro_trend,
        "swing_context_4h": swing_context,
        "micro_execution_15m": micro_exec,
        "recent_15m_ticks_history": history_list,
    }
    return json.dumps(ai_payload, indent=2)


def run_historical_orchestration(
    df_1d_tech, df_15m_tech, backtester, test_window=20, provider="openai", model_name=None,
    timeframe="15m", df_4h_tech=None, analysis="hybrid-spy"
):
    """
    Simulates live execution by walking forward through historical data,
    calling the AI for each bar and logging the result in the backtester.
    """
    print(f"\n{'=' * 55}")
    mode_labels = {"day": "Day", "4h": "4-Hour", "15m": "15-Min"}
    mode_label = mode_labels.get(timeframe, timeframe)
    print(f"  STARTING AI BACKTEST LOOP  |  mode={mode_label}  |  analysis={analysis}  |  window={test_window} bars")
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
            ai_payload = serialize_for_ai_day(current_slice, analysis)
        elif timeframe == "4h":
            ai_payload = serialize_for_ai_4h(df_1d_tech, current_slice, analysis)
        else:
            ai_payload = serialize_for_ai(df_1d_tech, df_4h_tech, current_slice, analysis)

        print(f"[{current_timestamp}] Querying {provider.upper()}...")
        try:
            raw_response = get_ai_market_signal(
                ai_payload, provider=provider, model_name=model_name,
                timeframe=timeframe, analysis=analysis
            )

            if "BUY" in raw_response.upper():
                signal = "BUY"
            elif "SELL" in raw_response.upper():
                signal = "SELL"
            else:
                signal = "HOLD"

            match = re.search(r"###\s*Confidence\s*\n.*?(\d+(?:\.\d+)?)\s*%", raw_response, re.IGNORECASE | re.DOTALL)
            confidence = f"{match.group(1)}%" if match else "N/A"

        except Exception as e:
            print(f"  API error: {e} — defaulting to HOLD")
            signal = "HOLD"
            confidence = "N/A"

        print(f"  Signal: {signal}  |  Confidence: {confidence}")

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
