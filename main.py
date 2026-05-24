"""
SP500 + VIX AI Trading Signal System
=====================================
Usage:
  python main.py predict   [--provider openai|deepseek|qwen] [--model MODEL]
  python main.py backtest  [--provider openai|deepseek|qwen] [--model MODEL]
                           [--test-window N] [--lookahead N]

Environment variables required (depending on provider):
  OPENAI_API_KEY
  DEEPSEEK_API_KEY
  DASHSCOPE_API_KEY
"""

import argparse
import json

from data_utils import fetch_and_prepare_data
from data_postpreposs import serialize_for_ai, run_historical_orchestration
from ai_agent_connector import get_ai_market_signal
from backtesting import MarketAIBacktester


def run_predict(provider, model_name):
    print("=" * 55)
    print("  SP500/VIX AI PREDICTION — CURRENT SIGNAL")
    print("=" * 55)

    df_1d_tech, df_15m_tech = fetch_and_prepare_data()

    payload = serialize_for_ai(df_1d_tech, df_15m_tech.dropna())
    response = get_ai_market_signal(payload, provider=provider, model_name=model_name)

    print("\n" + response)


def run_backtest(provider, model_name, test_window, lookahead):
    print("=" * 55)
    print("  SP500/VIX AI BACKTEST MODE")
    print(f"  Provider: {provider.upper()}  |  Test window: {test_window} bars  |  Lookahead: {lookahead} bars")
    print("=" * 55)

    df_1d_tech, df_15m_tech = fetch_and_prepare_data()
    df_15m_clean = df_15m_tech.dropna()

    if len(df_15m_clean) < test_window + lookahead:
        print(
            f"Error: only {len(df_15m_clean)} clean bars available, "
            f"but test_window ({test_window}) + lookahead ({lookahead}) = {test_window + lookahead}."
        )
        return

    backtester = MarketAIBacktester(df_15m_clean, lookahead_periods=lookahead)

    run_historical_orchestration(
        df_1d_tech=df_1d_tech,
        df_15m_tech=df_15m_clean,
        backtester=backtester,
        test_window=test_window,
        provider=provider,
        model_name=model_name,
    )


def main():
    parser = argparse.ArgumentParser(
        description="SP500/VIX AI Trading Signal System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("mode", choices=["predict", "backtest"], help="Operation mode")
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
        default=100,
        help="Bars ahead to verify signal accuracy (default: 100)",
    )

    args = parser.parse_args()

    if args.mode == "predict":
        run_predict(args.provider, args.model)
    else:
        run_backtest(args.provider, args.model, args.test_window, args.lookahead)


if __name__ == "__main__":
    main()
