import pandas as pd
import numpy as np

class MarketAIBacktester:
    def __init__(self, df, lookahead_periods=16):
        """
        Args:
            df (DataFrame): Historical futures data with technical indicators (daily or 15m).
            lookahead_periods (int): How many bars ahead to verify signal accuracy.
        """
        self.df = df.copy()
        self.lookahead = lookahead_periods
        self.results_log = []

    def evaluate_signal(self, timestamp, ai_signal):
        """
        Evaluates a single AI signal against what actually happened next in the market.
        """
        if timestamp not in self.df.index:
            return None
        
        # Get the row index of the signal timestamp
        idx = self.df.index.get_loc(timestamp)
        
        # Ensure we have enough future data rows to evaluate the lookahead window
        if idx + self.lookahead >= len(self.df):
            return None # Not enough future data yet (edge of the dataset)
            
        entry_price = self.df.iloc[idx]['SP500_Futures']
        future_price = self.df.iloc[idx + self.lookahead]['SP500_Futures']
        
        # Calculate the actual percentage change over the window
        price_change_pct = ((future_price - entry_price) / entry_price) * 100
        
        # Determine if the AI was correct
        is_correct = False
        trade_return = 0.0
        
        if ai_signal == "BUY":
            is_correct = price_change_pct > 0
            trade_return = price_change_pct
        elif ai_signal == "SELL":
            is_correct = price_change_pct < 0
            trade_return = -price_change_pct # Short trade profit when price drops
        elif ai_signal == "HOLD":
            # Hold is considered accurate if the market remained relatively flat (e.g., within +/- 0.1%)
            is_correct = abs(price_change_pct) <= 0.1
            trade_return = 0.0

        result = {
            "timestamp": timestamp,
            "signal": ai_signal,
            "entry_price": round(entry_price, 2),
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
        
        # Filter out 'HOLD' signals to isolate active trades
        active_trades = log_df[log_df['signal'].isin(['BUY', 'SELL'])]
        
        total_signals = len(log_df)
        total_active = len(active_trades)
        
        # 1. Calculate General Directional Accuracy
        correct_signals = log_df['is_correct'].sum()
        overall_accuracy = (correct_signals / total_signals) * 100 if total_signals > 0 else 0
        
        # 2. Calculate Active Trade Win Rate
        winning_trades = active_trades['is_correct'].sum()
        win_rate = (winning_trades / total_active) * 100 if total_active > 0 else 0
        
        # 3. Profit Factor (Gross Profits / Gross Losses)
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

# =====================================================================
# Demonstration: Simulating a log of past AI Decisions to test the module
# =====================================================================
if __name__ == "__main__":
    # Assuming df_15m_tech exists from your prior step:
    # backtester = MarketAIBacktester(df_15m_tech, lookahead_periods=16)
    
    print("Backtester engine loaded successfully.")
    print("Example framework setup: Feed timestamps and AI outputs into backtester.evaluate_signal()")