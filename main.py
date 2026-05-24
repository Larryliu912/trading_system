"""
SP500 + VIX AI Trading Signal System
=====================================
Usage:
  python main.py predict   [--timeframe day|4h|15m] [--provider openai|deepseek|qwen] [--model MODEL]
  python main.py backtest  [--timeframe day|4h|15m] [--provider openai|deepseek|qwen] [--model MODEL]
                           [--test-window N] [--lookahead N]

Timeframes:
  day   Analyze daily bars only → predict next trading day's direction
  4h    Analyze daily + 4-hour bars → predict next 4-hour bar's direction
  15m   Analyze daily + 15-min bars → predict next 15-min bar's direction (default)

Lookahead defaults: 1 bar (day), 4 bars (4h), 100 bars (15m) — override with --lookahead.

Environment variables required (depending on provider):
  OPENAI_API_KEY
  DEEPSEEK_API_KEY
  DASHSCOPE_API_KEY
"""

import argparse
import json

from data_utils import fetch_and_prepare_data
from data_postpreposs import serialize_for_ai, serialize_for_ai_day, serialize_for_ai_4h, run_historical_orchestration
from ai_agent_connector import get_ai_market_signal
from backtesting import MarketAIBacktester


def run_predict(provider, model_name, timeframe):
    print("=" * 55)
    labels = {"day": "NEXT DAY", "4h": "NEXT 4H", "15m": "NEXT 15-MIN"}
    print(f"  SP500/VIX AI PREDICTION — {labels.get(timeframe, timeframe)} SIGNAL")
    print("=" * 55)

    df_1d_tech, df_4h_tech, df_15m_tech = fetch_and_prepare_data(timeframe)

    if timeframe == "day":
        payload = serialize_for_ai_day(df_1d_tech.dropna())
    elif timeframe == "4h":
        payload = serialize_for_ai_4h(df_1d_tech, df_4h_tech.dropna())
    else:
        payload = serialize_for_ai(df_1d_tech, df_4h_tech, df_15m_tech.dropna())

    response = get_ai_market_signal(payload, provider=provider, model_name=model_name, timeframe=timeframe)
    print("\n" + response)


def run_backtest(provider, model_name, test_window, lookahead, timeframe):
    print("=" * 55)
    print("  SP500/VIX AI BACKTEST MODE")
    mode_labels = {"day": "Day", "4h": "4-Hour", "15m": "15-Min"}
    mode_label = mode_labels.get(timeframe, timeframe)
    print(f"  Mode: {mode_label}  |  Provider: {provider.upper()}  |  Test window: {test_window} bars  |  Lookahead: {lookahead} bars")
    print("=" * 55)

    df_1d_tech, df_4h_tech, df_15m_tech = fetch_and_prepare_data(timeframe)
    df_walk = {"day": df_1d_tech, "4h": df_4h_tech, "15m": df_15m_tech}[timeframe]
    df_clean = df_walk.dropna()
    bar_label = {"day": "daily", "4h": "4h", "15m": "15m"}.get(timeframe, timeframe)

    if len(df_clean) < test_window + lookahead:
        print(
            f"Error: only {len(df_clean)} clean {bar_label} bars available, "
            f"but test_window ({test_window}) + lookahead ({lookahead}) = {test_window + lookahead}."
        )
        return

    backtester = MarketAIBacktester(df_clean, lookahead_periods=lookahead)

    gaps = backtester.df.index.to_series().diff().dt.days.dropna()
    large_gaps = gaps[gaps > 5]
    if not large_gaps.empty:
        print(f"  GAPS > 5 days found:")
        for date, gap in large_gaps.items():
            print(f"    {date}: gap of {int(gap)} days before this date")
    else:
        print(f"  No large gaps found in backtester.df")
    print(f"------------------\n")

    run_historical_orchestration(
        df_1d_tech=df_1d_tech,
        df_15m_tech=df_15m_tech,
        df_4h_tech=df_4h_tech,
        backtester=backtester,
        test_window=test_window,
        provider=provider,
        model_name=model_name,
        timeframe=timeframe,
    )


def main():
    parser = argparse.ArgumentParser(
        description="SP500/VIX AI Trading Signal System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("mode", choices=["predict", "backtest"], help="Operation mode")
    parser.add_argument(
        "--timeframe",
        default="15m",
        choices=["day", "4h", "15m"],
        help="Analysis timeframe: 'day' (next-day), '4h' (next 4-hour), '15m' (next 15-min, default)",
    )
    parser.add_argument(
        "--provider",
        default="openai",
        choices=["openai", "deepseek", "qwen"],
        help="AI provider (default: openai)",
    )
    parser.add_argument("--model", default=None, help="Override the provider's default model")
    parser.add_argument(
        "--test-window",
        type=int,
        default=20,
        help="Bars to test in backtest mode (default: 20)",
    )
    parser.add_argument(
        "--lookahead",
        type=int,
        default=None,
        help="Bars ahead to verify signal accuracy (default: 1=day, 4=4h, 100=15m)",
    )

    args = parser.parse_args()

    if args.lookahead is None:
        args.lookahead = {"day": 1, "4h": 4, "15m": 100}.get(args.timeframe, 100)

    if args.mode == "predict":
        run_predict(args.provider, args.model, args.timeframe)
    else:
        run_backtest(args.provider, args.model, args.test_window, args.lookahead, args.timeframe)


if __name__ == "__main__":
    main()
