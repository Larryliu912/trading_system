"""
SP500 + VIX AI Trading Signal System
=====================================
Usage:
  python main.py predict   [--timeframe day|4h|15m] [--analysis pure-spy|pure-vix|hybrid-spy|hybrid-vix]
                           [--provider openai|deepseek|qwen] [--model MODEL]
  python main.py backtest  [--timeframe day|4h|15m] [--analysis pure-spy|pure-vix|hybrid-spy|hybrid-vix]
                           [--provider openai|deepseek|qwen] [--model MODEL]
                           [--test-window N] [--lookahead N]

Timeframes:
  day   Analyze daily bars only → predict next trading day's direction
  4h    Analyze daily + 4-hour bars → predict next 4-hour bar's direction
  15m   Analyze daily + 15-min bars → predict next 15-min bar's direction (default)

Analysis modes:
  pure-spy   Use only SP500 data to predict SP500 direction
  pure-vix   Use only VIX data to predict VIX direction
  hybrid-spy Use SP500 + VIX data to predict SP500 direction (default)
  hybrid-vix Use SP500 + VIX data to predict VIX direction

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

_ANALYSIS_LABELS = {
    "pure-spy": "Pure SP500",
    "pure-vix": "Pure VIX",
    "hybrid-spy": "Hybrid→SP500",
    "hybrid-vix": "Hybrid→VIX",
}


def run_predict(provider, model_name, timeframe="day", analysis="hybrid-spy"):
    print("=" * 60)
    tf_labels = {"day": "NEXT DAY", "4h": "NEXT 4H", "15m": "NEXT 15-MIN"}
    print(f"  AI PREDICTION — {tf_labels.get(timeframe, timeframe)} | {_ANALYSIS_LABELS.get(analysis, analysis)}")
    print("=" * 60)

    df_1d_tech, df_4h_tech, df_15m_tech = fetch_and_prepare_data(timeframe, analysis)

    if timeframe == "day":
        payload = serialize_for_ai_day(df_1d_tech.dropna(), analysis)
    elif timeframe == "4h":
        payload = serialize_for_ai_4h(df_1d_tech, df_4h_tech.dropna(), analysis)
    else:
        payload = serialize_for_ai(df_1d_tech, df_4h_tech, df_15m_tech.dropna(), analysis)

    response = get_ai_market_signal(payload, provider=provider, model_name=model_name,
                                    timeframe=timeframe, analysis=analysis)
    print("\n" + response)


def run_backtest(provider, model_name, test_window=20, lookahead=30, timeframe='day', analysis='hybrid-spy'):
    print("=" * 60)
    print("  AI BACKTEST MODE")
    mode_label = {"day": "Day", "4h": "4-Hour", "15m": "15-Min"}.get(timeframe, timeframe)
    print(
        f"  Timeframe: {mode_label}  |  Analysis: {_ANALYSIS_LABELS.get(analysis, analysis)}"
        f"  |  Provider: {provider.upper()}  |  Test window: {test_window} bars  |  Lookahead: {lookahead} bars"
    )
    print("=" * 60)

    df_1d_tech, df_4h_tech, df_15m_tech = fetch_and_prepare_data(timeframe, analysis)
    df_walk = {"day": df_1d_tech, "4h": df_4h_tech, "15m": df_15m_tech}[timeframe]
    df_clean = df_walk.dropna()
    bar_label = {"day": "daily", "4h": "4h", "15m": "15m"}.get(timeframe, timeframe)

    if len(df_clean) < test_window + lookahead:
        print(
            f"Error: only {len(df_clean)} clean {bar_label} bars available, "
            f"but test_window ({test_window}) + lookahead ({lookahead}) = {test_window + lookahead}."
        )
        return

    target_col = "VIX_Futures" if analysis in ("pure-vix", "hybrid-vix") else "SP500_Futures"
    backtester = MarketAIBacktester(df_clean, lookahead_periods=lookahead, target_column=target_col)

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
        analysis=analysis,
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
        help="Analysis timeframe: 'day', '4h', or '15m' (default)",
    )
    parser.add_argument(
        "--analysis",
        default="hybrid-spy",
        choices=["pure-spy", "pure-vix", "hybrid-spy", "hybrid-vix"],
        help=(
            "Analysis mode — what data to use and what to predict: "
            "'pure-spy' (SP500 only→SP500), 'pure-vix' (VIX only→VIX), "
            "'hybrid-spy' (SP500+VIX→SP500, default), 'hybrid-vix' (SP500+VIX→VIX)"
        ),
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
        run_predict(args.provider, args.model, args.timeframe, args.analysis)
    else:
        run_backtest(args.provider, args.model, args.test_window, args.lookahead, args.timeframe, args.analysis)


if __name__ == "__main__":
    main()
