# AI Trading & Stock Research System

An AI-powered toolkit for market analysis across two modes:

1. **Single Stock Research** — deep five-layer fundamental analysis, short-term technical analysis, and hyperscaler AI CAPEX tracking for individual equities
2. **SP500 / VIX Signal Engine** — multi-timeframe BUY / SELL / HOLD signals for S&P 500 and VIX futures

A Flask web UI ties everything together with streaming output, a report library, and scheduled recurring runs.

---

## Project Structure

```
sp500index/
├── single_stock_research/
│   ├── main.py                  # CLI entry point for all stock research modes
│   ├── iv_history.json          # Persisted LEAP IV snapshots (90-day rolling)
│   └── reports/
│       ├── long_term/           # Five-layer analysis reports
│       ├── short_term/          # Short-term technical analysis reports
│       └── hyperscaler/         # Hyperscaler AI CAPEX reports
├── common/
│   ├── data_utils.py            # SP500 / VIX data fetching and indicator computation
│   └── data_postpreposs.py      # Serializes market state → AI JSON payload
├── web_ui/
│   └── app.py                   # Flask UI — run analyses, stream output, schedule jobs
├── ai_agent_connector.py        # SP500 / VIX AI signal connector
└── prompts.py                   # All system prompts
```

---

## Single Stock Research

`single_stock_research/main.py` supports three analysis modes via flags.

### Five-Layer Long-Term Analysis (default)

The AI works autonomously through five layers using tool calling — it decides what data to fetch and when. No data is pre-loaded.

```
Layer 1 · Macro        — economic cycle, rates, inflation, risk premium
Layer 2 · Industry     — lifecycle, TAM, competition, policy
Layer 3 · Fundamentals — business model, moat, financials, growth
Layer 4 · Valuation    — P/E history, scenarios, margin of safety
Layer 5 · Decision     — position sizing, catalysts, risks, bear case
```

**Tools available to the AI:**

| Tool | Returns |
|---|---|
| `get_price_history` | Monthly closes, 1m–3y returns, 52-week range, beta, analyst target |
| `get_pe_history` | Trailing P/E series, forward P/E, PEG, historical percentile |
| `get_financials` | Annual + quarterly income statement, margins, FCF, balance sheet |
| `get_company_info` | Sector, industry, market cap, business description |
| `get_macro_indicators` | VIX, 10Y/2Y yields, DXY, SPY 1-year return |
| `get_recent_news` | Last 12 months of Yahoo Finance headlines |

**Usage:**

```bash
# Basic
python single_stock_research/main.py NVDA

# With provider and portfolio context
python single_stock_research/main.py TSLA --provider claude --portfolio "AAPL 30%, cash 50%"

# Override model
python single_stock_research/main.py MSFT --provider openai --model gpt-4o
```

Output is saved to `single_stock_research/reports/long_term/TICKER_YYYYMMDD_HHMMSS_PROVIDER.md`.

---

### Short-Term Technical Analysis (`--short-term`)

Fetches 15-minute OHLCV bars, near-term option chains, and LEAP IV — then generates a directional bias (long / short / neutral) with entry, target, and stop levels. Analysis is written in Chinese.

**Tools available to the AI:**

| Tool | Returns |
|---|---|
| `get_short_term_data` | 15m OHLCV + EMA(9/21/50), RSI(14), MACD, Bollinger Bands, ATR(14), VWAP |
| `get_option_chain` | ATM IV, put/call ratios, max pain, top OI strikes for nearest expirations |
| `get_leap_iv` | LEAP IV at 6m and 1y expirations, ±10% OTM skew, IV trend vs 1w/1m history |
| `get_price_history` | Daily chart for broader trend context |

**Usage:**

```bash
python single_stock_research/main.py NVDA --short-term
python single_stock_research/main.py MU --short-term --provider deepseek
```

Output is saved to `single_stock_research/reports/short_term/TICKER_YYYYMMDD_HHMMSS_PROVIDER_short_term.md`.

**LEAP IV history** is automatically persisted to `single_stock_research/iv_history.json` (90-day rolling window per ticker per tenor) and used to surface rising / stable / falling IV trends on every subsequent run.

---

### Hyperscaler AI CAPEX Analysis (`--hyperscaler`)

Fetches quarterly CAPEX, revenue, operating income, and recent news for **GOOGL / AMZN / MSFT / META** to answer two questions:

1. Is AI CAPEX across the sector expanding or compressing?
2. Is AI investment translating into revenue and income growth, or is it a drag?

```bash
python single_stock_research/main.py --hyperscaler
python single_stock_research/main.py --hyperscaler --provider claude
```

Output is saved to `single_stock_research/reports/hyperscaler/hyperscaler_YYYYMMDD_HHMMSS_PROVIDER.md`.

A recent hyperscaler report (≤ 7 days old) is automatically injected as context into the five-layer macro layer, so long-term analyses stay current on AI CAPEX trends without re-fetching the data.

---

## SP500 / VIX Signal Engine

The signal engine fetches live ES=F and/or ^VIX data, computes technical indicators, and sends a compact JSON payload to an LLM to generate a directional signal.

**Architecture:**

```
common/data_utils.py          — yfinance → ES=F / ^VIX bars + MA/BB/RSI/MACD
common/data_postpreposs.py    — compresses macro + micro state → JSON payload
ai_agent_connector.py         — sends payload to LLM, returns BIAS · SIGNAL
```

**Analysis modes** (`--analysis` flag):

| Mode | Data Used | Predicts |
|---|---|---|
| `pure-spy` | SP500 only | SP500 direction |
| `pure-vix` | VIX only | VIX direction |
| `hybrid-spy` | SP500 + VIX | SP500 direction *(default)* |
| `hybrid-vix` | SP500 + VIX | VIX direction |

**Timeframes** (`--timeframe` flag):

| Timeframe | Data Tiers | Predicts |
|---|---|---|
| `day` | Daily bars | Next trading day |
| `4h` | Daily + 4h bars | Next 4-hour bar |
| `15m` | Daily + 4h + 15m bars | Next 15-minute bar *(default)* |

**Technical indicators** (computed on the prediction-target column):

| Indicator | Parameters |
|---|---|
| Simple Moving Averages | 20 / 60 / 90 periods |
| Bollinger Bands | 20-period, ±2σ |
| RSI | 14-period (Wilder's EMA) |
| MACD | 12 / 26 EMA, 9 signal |

**Data sources:**

| Asset | Ticker | Fallback |
|---|---|---|
| S&P 500 Futures | `ES=F` | — |
| VIX Index | `^VIX` | `VXX` (intraday only) |

---

## Web UI

Start the Flask app and open `http://127.0.0.1:5000` in a browser.

```bash
python web_ui/app.py
```

**Features:**
- Run long-term or short-term analyses for any ticker, with live streaming output
- Browse, read, and delete saved reports (long-term, short-term, hyperscaler)
- Schedule recurring analyses — daily or weekly at a specified time, for one or more tickers

---

## Supported AI Providers

All providers use the OpenAI-compatible chat completions API. Switching requires only the `--provider` flag.

| Provider | Env Variable | Default Model |
|---|---|---|
| OpenAI | `OPENAI_API_KEY` | `gpt-4o` |
| DeepSeek | `DEEPSEEK_API_KEY` | `deepseek-v4-pro` |
| Qwen (Alibaba DashScope) | `DASHSCOPE_API_KEY` | `qwen-plus` *(default)* |
| Claude (Anthropic) | `ANTHROPIC_API_KEY` | `claude-opus-4-7` |

---

## Requirements

- Python 3.10+
- API key for at least one provider

```bash
pip install yfinance openai pandas numpy flask
```

---

## Configuration

Set API keys as environment variables before running.

**Windows (PowerShell):**
```powershell
$env:OPENAI_API_KEY    = "sk-..."
$env:DEEPSEEK_API_KEY  = "..."
$env:DASHSCOPE_API_KEY = "..."
$env:ANTHROPIC_API_KEY = "..."
```

**macOS / Linux:**
```bash
export OPENAI_API_KEY="sk-..."
export DEEPSEEK_API_KEY="..."
export DASHSCOPE_API_KEY="..."
export ANTHROPIC_API_KEY="..."
```
