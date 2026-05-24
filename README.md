# SP500 + VIX AI Trading Signal System

An AI-powered multi-timeframe analysis system for S&P 500 and VIX futures. It fetches live market data, computes technical indicators, and prompts an external LLM to generate directional trading signals (BUY / SELL / HOLD). A built-in backtester can replay historical bars and measure how often the AI's signals were correct.

---

## How It Works

```
                        ┌─────────────────────────────────────────┐
                        │               main.py                   │
                        │          predict | backtest             │
                        └────────────────┬────────────────────────┘
                                         │
              ┌──────────────────────────▼──────────────────────────┐
              │                    data_utils.py                    │
              │   yfinance → ES=F (SP500) + ^VIX (VIX Index)        │
              │   Timeframes: 15-minute (60 days) + Daily (1 year)  │
              │   Indicators: MA20/60/90, Bollinger Bands, RSI, MACD│
              └──────────────────────────┬──────────────────────────┘
                                         │
              ┌──────────────────────────▼──────────────────────────┐
              │                 data_postpreposs.py                 │
              │   Compresses macro + micro state → JSON payload     │
              │   (recent 5 ticks, Bollinger position %, bias tag)  │
              └──────────────────────────┬──────────────────────────┘
                                         │
              ┌──────────────────────────▼──────────────────────────┐
              │               ai_agent_connector.py                 │
              │      OpenAI / DeepSeek / Qwen (DashScope)           │
              │      Returns: BIAS · VOLATILITY · SIGNAL            │
              └──────────────────────────┬──────────────────────────┘
                                         │
                           (backtest mode only)
                                         │
              ┌──────────────────────────▼──────────────────────────┐
              │                   backtesting.py                    │
              │   Verifies each signal N bars into the future        │
              │   Reports: Win Rate, Profit Factor, Total Return     │
              └─────────────────────────────────────────────────────┘
```

---

## Project Structure

| File | Role |
|---|---|
| `main.py` | Entry point — CLI argument parsing, orchestrates predict and backtest modes |
| `data_utils.py` | Fetches market data from Yahoo Finance and computes all technical indicators |
| `data_postpreposs.py` | Serializes processed data into a token-efficient JSON payload for the LLM |
| `ai_agent_connector.py` | Sends the payload to the configured AI provider and returns the raw analysis |
| `backtesting.py` | Evaluates AI signals against future price movement and computes performance metrics |

---

## Requirements

- Python 3.10+
- An API key for at least one supported AI provider

Install dependencies:

```bash
pip install yfinance openai pandas numpy
```

---

## Configuration

Set your API key as an environment variable before running. Only the key for the provider you use is required.

**Windows (PowerShell):**
```powershell
$env:OPENAI_API_KEY   = "sk-..."
$env:DEEPSEEK_API_KEY = "..."
$env:DASHSCOPE_API_KEY = "..."   # Qwen / Alibaba DashScope
```

**macOS / Linux:**
```bash
export OPENAI_API_KEY="sk-..."
export DEEPSEEK_API_KEY="..."
export DASHSCOPE_API_KEY="..."
```

---

## Usage

### Predict — get a live signal right now

Fetches the latest market data and returns one BUY / SELL / HOLD signal.

```bash
# OpenAI (default)
python main.py predict --provider openai

# DeepSeek
python main.py predict --provider deepseek

# Qwen with a specific model
python main.py predict --provider qwen --model qwen-max
```

**Example output:**
```
### BIAS ASSESSMENT
Macro trend is bullish (ES=F above 90-day MA). 15m momentum is weakening —
RSI at 58 and MACD histogram contracting.

### VOLATILITY CHECK
VIX is flat at 13.2. No spike or mean-reversion signal present; volatility
is not contradicting the directional bias.

### STRATEGIC SIGNAL
BUY — Structural uptrend intact with consolidation on the 15m providing a
low-risk re-entry point.
```

---

### Backtest — measure historical accuracy

Replays historical bars one at a time, asks the AI for a signal at each bar, then checks whether the price moved in the predicted direction within the lookahead window.

```bash
# Default: test last 20 bars, verify each signal 100 bars (~25 hours) ahead
python main.py backtest --provider openai

# Deeper test: 50 bars, 16-bar lookahead (~4 hours)
python main.py backtest --provider openai --test-window 50 --lookahead 16

# Use DeepSeek to reduce cost on large windows
python main.py backtest --provider deepseek --test-window 100 --lookahead 100
```

| Flag | Default | Description |
|---|---|---|
| `--test-window` | `20` | Number of historical 15m bars to generate signals for |
| `--lookahead` | `100` | Bars ahead (~25 hours) used to verify each signal |

**Example output:**
```
=======================================================
  BACKTEST COMPLETE — FINAL METRICS
=======================================================
{
    "Summary": {
        "Total Signals Generated": 20,
        "Active Trades (BUY/SELL)": 17,
        "Overall System Accuracy": "65.00%",
        "Active Trade Win Rate": "70.59%"
    },
    "Financials": {
        "Total Compounded Return": "4.821%",
        "Gross Profits Sum": "6.103%",
        "Gross Losses Sum": "1.282%",
        "Profit Factor": 4.76
    }
}
```

> **Note:** Each bar triggers one API call. A `--test-window 20` run makes 20 calls. Factor in your provider's rate limits and token costs for large windows.

---

## Technical Indicators

All indicators are calculated on the S&P 500 price column of each timeframe.

| Indicator | Parameters | Used for |
|---|---|---|
| Simple Moving Averages | 20 / 60 / 90 periods | Macro structural bias |
| Bollinger Bands | 20-period, ±2σ | Micro execution context |
| RSI | 14-period (Wilder's EMA) | Momentum overbought/oversold |
| MACD | 12 / 26 EMA, 9 signal | Momentum direction and histogram |

---

## Data Sources

| Data | Ticker | Interval | Lookback |
|---|---|---|---|
| S&P 500 Futures | `ES=F` | 15m + 1d | 60 days + 1 year |
| VIX Index | `^VIX` (fallback: `VXX`) | 15m + 1d | 60 days + 1 year |

Data is fetched live from Yahoo Finance via `yfinance` each run. For the 15m VIX feed, `^VIX` is tried first; if Yahoo Finance does not serve intraday index data, `VXX` (VIX Short-Term Futures ETN) is used automatically.

---

## Supported AI Providers

| Provider | Env Variable | Default Model |
|---|---|---|
| OpenAI | `OPENAI_API_KEY` | `gpt-4o` |
| DeepSeek | `DEEPSEEK_API_KEY` | `deepseek-chat` |
| Qwen (Alibaba) | `DASHSCOPE_API_KEY` | `qwen-plus` |

All providers use the OpenAI-compatible chat completions API, so switching between them requires only a `--provider` flag change.
