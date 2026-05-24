# SP500 + VIX AI Trading Signal System

An AI-powered multi-timeframe analysis system for S&P 500 and VIX futures. It fetches live market data, computes technical indicators, and prompts an external LLM to generate directional trading signals (BUY / SELL / HOLD). A built-in backtester can replay historical bars and measure how often the AI's signals were correct.

---

## How It Works

```
                        ┌─────────────────────────────────────────┐
                        │               main.py                   │
                        │   predict | backtest                    │
                        │   --timeframe  day | 4h | 15m           │
                        │   --analysis   pure-spy | pure-vix      │
                        │               hybrid-spy | hybrid-vix   │
                        └────────────────┬────────────────────────┘
                                         │
              ┌──────────────────────────▼──────────────────────────┐
              │                    data_utils.py                    │
              │   yfinance → ES=F and/or ^VIX (per analysis mode)   │
              │   Timeframes: 1d (1 year) · 4h (180 days)           │
              │                         · 15m (60 days)             │
              │   Indicators: MA20/60/90, Bollinger Bands, RSI, MACD│
              │   (computed on the prediction-target column)        │
              └──────────────────────────┬──────────────────────────┘
                                         │
              ┌──────────────────────────▼──────────────────────────┐
              │                 data_postpreposs.py                 │
              │   Compresses macro + micro state → JSON payload     │
              │   Includes only the assets relevant to the mode     │
              └──────────────────────────┬──────────────────────────┘
                                         │
              ┌──────────────────────────▼──────────────────────────┐
              │               ai_agent_connector.py                 │
              │      OpenAI / DeepSeek / Qwen (DashScope)           │
              │      System prompt adapts to analysis mode          │
              │      Returns: BIAS · VOLATILITY · SIGNAL            │
              └──────────────────────────┬──────────────────────────┘
                                         │
                           (backtest mode only)
                                         │
              ┌──────────────────────────▼──────────────────────────┐
              │                   backtesting.py                    │
              │   Verifies each signal N bars into the future        │
              │   Evaluates against SP500 or VIX (per mode)         │
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
$env:OPENAI_API_KEY    = "sk-..."
$env:DEEPSEEK_API_KEY  = "..."
$env:DASHSCOPE_API_KEY = "..."   # Qwen / Alibaba DashScope
```

**macOS / Linux:**
```bash
export OPENAI_API_KEY="sk-..."
export DEEPSEEK_API_KEY="..."
export DASHSCOPE_API_KEY="..."
```

---

## Analysis Modes

The `--analysis` flag controls what data is fetched and what the AI is asked to predict.

| Mode | Data Used | Predicts |
|---|---|---|
| `pure-spy` | SP500 only | SP500 direction |
| `pure-vix` | VIX only | VIX direction |
| `hybrid-spy` | SP500 + VIX | SP500 direction *(default)* |
| `hybrid-vix` | SP500 + VIX | VIX direction |

- **Pure modes** fetch a single asset and omit the other from the payload entirely, giving the AI a clean single-asset signal with no cross-asset noise.
- **Hybrid modes** provide both SP500 and VIX data so the AI can exploit their inverse correlation. `hybrid-spy` predicts where SP500 goes; `hybrid-vix` predicts where VIX goes.
- Technical indicators (MA, Bollinger Bands, RSI, MACD) are always computed on the **prediction target** column — VIX for vix modes, SP500 for spy modes.

---

## Timeframes

The `--timeframe` flag controls the data granularity and the prediction horizon.

| Timeframe | Data Tiers | Predicts |
|---|---|---|
| `day` | Daily bars only | Next trading day's direction |
| `4h` | Daily + 4-hour bars | Next 4-hour bar's direction |
| `15m` | Daily + 4h + 15-minute bars | Next 15-minute bar's direction *(default)* |

---

## Usage

### Predict — get a live signal right now

Fetches the latest market data and returns one BUY / SELL / HOLD signal.

```bash
# Default: hybrid SP500 prediction on 15m timeframe via OpenAI
python main.py predict

# Pure VIX prediction (daily timeframe)
python main.py predict --analysis pure-vix --timeframe day

# Hybrid: use SP500 + VIX data to predict VIX direction (4h timeframe)
python main.py predict --analysis hybrid-vix --timeframe 4h --provider deepseek

# Pure SP500 prediction via Qwen with a specific model
python main.py predict --analysis pure-spy --provider qwen --model qwen-max
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
BUY S&P 500 Futures — Structural uptrend intact with consolidation on the
15m providing a low-risk re-entry point.

### Confidence
72%
```

---

### Backtest — measure historical accuracy

Replays historical bars one at a time, asks the AI for a signal at each bar, then checks whether the price moved in the predicted direction within the lookahead window.

```bash
# Default: hybrid-spy, 15m timeframe, last 20 bars, 100-bar lookahead
python main.py backtest --provider openai

# Pure VIX backtest on daily bars
python main.py backtest --analysis pure-vix --timeframe day --test-window 30 --lookahead 1

# Hybrid VIX on 4h, wider window
python main.py backtest --analysis hybrid-vix --timeframe 4h --test-window 50 --lookahead 4

# Use DeepSeek to reduce cost on large windows
python main.py backtest --provider deepseek --test-window 100 --lookahead 100
```

| Flag | Default | Description |
|---|---|---|
| `--analysis` | `hybrid-spy` | Analysis mode (see table above) |
| `--timeframe` | `15m` | Data granularity and prediction horizon |
| `--test-window` | `20` | Number of historical bars to generate signals for |
| `--lookahead` | `1` (day) / `4` (4h) / `100` (15m) | Bars ahead used to verify each signal |

**Example output:**
```
============================================================
  AI BACKTEST MODE
  Timeframe: 15-Min  |  Analysis: Hybrid→SP500  |  Provider: OPENAI
  Test window: 20 bars  |  Lookahead: 100 bars
============================================================
...
============================================================
  BACKTEST COMPLETE — FINAL METRICS
============================================================
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

Indicators are calculated on the **prediction target** price column for each timeframe tier.

| Indicator | Parameters | Used for |
|---|---|---|
| Simple Moving Averages | 20 / 60 / 90 periods | Macro structural bias |
| Bollinger Bands | 20-period, ±2σ | Execution context and squeeze detection |
| RSI | 14-period (Wilder's EMA) | Momentum overbought/oversold |
| MACD | 12 / 26 EMA, 9 signal | Momentum direction and histogram |

---

## Data Sources

| Data | Ticker | Intervals | Lookback |
|---|---|---|---|
| S&P 500 Futures | `ES=F` | 15m · 1h · 1d | 60 days · 180 days · 1 year |
| VIX Index | `^VIX` (fallback: `VXX`) | 15m · 1h · 1d | 60 days · 180 days · 1 year |

Data is fetched live from Yahoo Finance via `yfinance` each run. For intraday VIX data, `^VIX` is tried first; if Yahoo Finance does not serve intraday index data, `VXX` (VIX Short-Term Futures ETN) is used automatically. In pure modes, only the relevant asset is fetched.

---

## Supported AI Providers

| Provider | Env Variable | Default Model |
|---|---|---|
| OpenAI | `OPENAI_API_KEY` | `gpt-4o` |
| DeepSeek | `DEEPSEEK_API_KEY` | `deepseek-chat` |
| Qwen (Alibaba) | `DASHSCOPE_API_KEY` | `qwen-plus` |

All providers use the OpenAI-compatible chat completions API, so switching between them requires only a `--provider` flag change. The system prompt is automatically tailored to the active analysis mode and timeframe.
