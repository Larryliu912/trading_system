import pandas as pd
import numpy as np

class MarketAIBacktester:
    def __init__(self, df, lookahead_periods=16, target_column="SP500_Futures"):
        """
        Args:
            df (DataFrame): Historical market data with technical indicators.
            lookahead_periods (int): How many bars ahead to verify signal accuracy.
            target_column (str): Price column to evaluate signals against.
        """
        self.df = df.copy()
        self.lookahead = lookahead_periods
        self.target_column = target_column
        self.results_log = []

    def evaluate_signal(self, timestamp, ai_signal):
        """
        Evaluates a single AI signal against what actually happened next in the market.
        """
        if timestamp not in self.df.index:
            return None

        idx = self.df.index.get_loc(timestamp)

        if idx + self.lookahead >= len(self.df):
            return None

        entry_price = self.df.iloc[idx][self.target_column]
        future_idx = idx + 1
        future_price = self.df.iloc[future_idx][self.target_column]
        future_timestamp = self.df.index[future_idx]

        price_change_pct = ((future_price - entry_price) / entry_price) * 100

        is_correct = False
        trade_return = 0.0

        if ai_signal == "BUY":
            is_correct = price_change_pct > 0
            trade_return = price_change_pct
        elif ai_signal == "SELL":
            is_correct = price_change_pct < 0
            trade_return = -price_change_pct
        elif ai_signal == "HOLD":
            is_correct = abs(price_change_pct) <= 0.1
            trade_return = 0.0

        result = {
            "timestamp": timestamp,
            "signal": ai_signal,
            "entry_price": round(entry_price, 2),
            "future_timestamp": future_timestamp,
            "future_price": round(future_price, 2),
            "price_change_pct": round(price_change_pct, 3),
            "trade_return_pct": round(trade_return, 3),
            "is_correct": is_correct
        }

        self.results_log.append(result)
        return result

    def calculate_performance_metrics(self):
        """
        Aggregates all evaluated signals into a final performance accuracy report.
        """
        if not self.results_log:
            return "No evaluated trades found in the log."

        log_df = pd.DataFrame(self.results_log)

        active_trades = log_df[log_df['signal'].isin(['BUY', 'SELL'])]

        total_signals = len(log_df)
        total_active = len(active_trades)

        correct_signals = log_df['is_correct'].sum()
        overall_accuracy = (correct_signals / total_signals) * 100 if total_signals > 0 else 0

        winning_trades = active_trades['is_correct'].sum()
        win_rate = (winning_trades / total_active) * 100 if total_active > 0 else 0

        gross_profits = active_trades[active_trades['trade_return_pct'] > 0]['trade_return_pct'].sum()
        gross_losses = abs(active_trades[active_trades['trade_return_pct'] < 0]['trade_return_pct'].sum())
        profit_factor = gross_profits / gross_losses if gross_losses > 0 else float('inf')

        report = {
            "Summary": {
                "Total Signals Generated": total_signals,
                "Active Trades (BUY/SELL)": total_active,
                "Overall System Accuracy": f"{overall_accuracy:.2f}%",
                "Active Trade Win Rate": f"{win_rate:.2f}%"
            },
            "Financials": {
                "Total Compounded Return": f"{active_trades['trade_return_pct'].sum():.3f}%",
                "Gross Profits Sum": f"{gross_profits:.3f}%",
                "Gross Losses Sum": f"{gross_losses:.3f}%",
                "Profit Factor": round(profit_factor, 2)
            }
        }
        return report


if __name__ == "__main__":
    print("Backtester engine loaded successfully.")
    print("Example framework setup: Feed timestamps and AI outputs into backtester.evaluate_signal()")
