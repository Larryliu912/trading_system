"""
Single Stock Research — Five-Layer AI Skill with Tool Calling
=============================================================
The AI works through five layers of analysis autonomously, calling data-fetch
functions whenever it needs real numbers. No data is pre-loaded; the model
decides what to fetch and when.

Five layers:
  Layer 1 · Macro          — economic cycle, rates, inflation, risk premium
  Layer 2 · Industry       — lifecycle, TAM, competition, policy
  Layer 3 · Fundamentals   — business model, moat, financials, growth
  Layer 4 · Valuation      — PE history, scenarios, margin of safety
  Layer 5 · Decision       — position sizing, catalysts, risks, bear case

Usage:
  python single_stock_research/main.py AAPL
  python single_stock_research/main.py NVDA --provider deepseek
  python single_stock_research/main.py NVDA --provider claude
  python single_stock_research/main.py TSLA --portfolio "AAPL 30%, MSFT 20%, cash 50%"

Environment variables:
  OPENAI_API_KEY / DEEPSEEK_API_KEY / DASHSCOPE_API_KEY / ANTHROPIC_API_KEY
"""

import argparse
import json
import os
import sys
import warnings
from datetime import datetime

import pandas as pd
import yfinance as yf
from openai import OpenAI

from prompts import FIVE_LAYER_SYSTEM_PROMPT

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Tool implementations  (called by the agent loop when AI requests them)
# ---------------------------------------------------------------------------

def _tz_strip(idx):
    return idx.tz_localize(None) if idx.tzinfo is not None else idx


def _round(v, n=2):
    try:
        return round(float(v), n)
    except Exception:
        return None


def _extract_eps(stmt) -> "pd.Series | None":
    for name in ("Diluted EPS", "Basic EPS"):
        if name in stmt.index:
            return stmt.loc[name].dropna().sort_index()
    return None


def get_price_history(ticker: str, years: int = 3) -> dict:
    """Return price history, key returns, 52-week range, beta, and analyst target."""
    t = yf.Ticker(ticker)
    start = (pd.Timestamp.today() - pd.DateOffset(years=years)).strftime("%Y-%m-%d")
    hist = t.history(start=start, auto_adjust=True)
    if hist.empty:
        return {"error": f"No price data for {ticker}"}

    price = hist["Close"]
    price.index = _tz_strip(price.index)

    def ret(n):
        return _round((price.iloc[-1] / price.iloc[-n] - 1) * 100) if len(price) >= n else None

    # Monthly OHLC for the chart context (last 24 months)
    monthly = price.resample("ME").last().tail(24)
    monthly_dict = {str(d.date()): _round(v) for d, v in monthly.items()}

    info = t.info
    return {
        "current_price": _round(price.iloc[-1]),
        "52w_high": _round(price[-252:].max()) if len(price) >= 252 else _round(price.max()),
        "52w_low":  _round(price[-252:].min()) if len(price) >= 252 else _round(price.min()),
        "returns_pct": {
            "1m":  ret(22),
            "3m":  ret(63),
            "6m":  ret(126),
            "1y":  ret(252),
            "3y":  ret(756),
        },
        "beta":               _round(info.get("beta")),
        "analyst_target":     _round(info.get("targetMeanPrice")),
        "analyst_buy_pct":    info.get("recommendationKey"),
        "monthly_close":      monthly_dict,
    }


def get_pe_history(ticker: str, years: int = 5) -> dict:
    """Return trailing P/E history built from annual + quarterly EPS."""
    t = yf.Ticker(ticker)
    parts = []
    try:
        a = _extract_eps(t.income_stmt)
        if a is not None and not a.empty:
            parts.append(a)
    except Exception:
        pass
    try:
        q = _extract_eps(t.quarterly_income_stmt)
        if q is not None and len(q) >= 4:
            parts.append(q.rolling(4).sum().dropna())
    except Exception:
        pass

    if not parts:
        return {"error": "No EPS data available"}

    ttm = pd.concat(parts).sort_index()
    ttm = ttm[~ttm.index.duplicated(keep="last")]

    start = (pd.Timestamp.today() - pd.DateOffset(years=years)).strftime("%Y-%m-%d")
    hist = t.history(start=start, auto_adjust=True)
    if hist.empty:
        return {"error": "No price data"}

    price = hist["Close"]
    price.index = _tz_strip(price.index)

    ttm_daily = ttm.reindex(price.index, method="ffill")
    valid = ttm_daily.notna() & (ttm_daily > 0)
    pe = price[valid] / ttm_daily[valid]

    if pe.empty:
        return {"error": "Could not compute P/E series"}

    current = _round(pe.iloc[-1])
    percentile = _round(float((pe <= current).mean() * 100)) if current else None

    quarterly = {
        str(d.date()): _round(v)
        for d, v in pe.resample("QE").last().dropna().tail(20).items()
    }

    return {
        "current_pe":           current,
        "forward_pe":           _round(t.info.get("forwardPE")),
        "peg_ratio":            _round(t.info.get("pegRatio")),
        "history_stats": {
            "median":           _round(pe.median()),
            "low":              _round(pe.min()),
            "high":             _round(pe.max()),
            "current_percentile": percentile,
        },
        "quarterly_pe":         quarterly,
    }


def get_financials(ticker: str) -> dict:
    """Return annual + recent quarterly income statement, margins, FCF, and balance sheet."""
    t = yf.Ticker(ticker)

    def stmt_row(stmt, *candidates):
        for c in candidates:
            if c in stmt.index:
                row = stmt.loc[c].dropna().sort_index()
                return {str(d.date()): _round(v / 1e6) for d, v in row.items()}
        return {}

    annual_rev  = stmt_row(t.income_stmt,           "Total Revenue")
    annual_gp   = stmt_row(t.income_stmt,           "Gross Profit")
    annual_op   = stmt_row(t.income_stmt,           "Operating Income", "EBIT")
    annual_ni   = stmt_row(t.income_stmt,           "Net Income")
    annual_eps  = {str(d.date()): _round(v) for d, v in
                   (_extract_eps(t.income_stmt) or pd.Series(dtype=float)).items()}

    q_rev  = stmt_row(t.quarterly_income_stmt, "Total Revenue")
    q_gp   = stmt_row(t.quarterly_income_stmt, "Gross Profit")
    q_ni   = stmt_row(t.quarterly_income_stmt, "Net Income")
    q_eps  = {str(d.date()): _round(v) for d, v in
              (_extract_eps(t.quarterly_income_stmt) or pd.Series(dtype=float)).items()}

    def margins(rev, profit):
        return {d: _round(profit[d] / rev[d] * 100) for d in rev if d in profit and rev[d]}

    gross_margin = margins(annual_rev, annual_gp)
    net_margin   = margins(annual_rev, annual_ni)
    op_margin    = margins(annual_rev, annual_op)

    # FCF = operating CF + capex (capex is negative in yfinance)
    fcf_M = None
    try:
        cf = t.cashflow
        op  = cf.loc["Operating Cash Flow"].iloc[0]  if "Operating Cash Flow"  in cf.index else None
        cap = cf.loc["Capital Expenditure"].iloc[0]   if "Capital Expenditure"   in cf.index else None
        if op is not None and cap is not None:
            fcf_M = _round((op + cap) / 1e6)
    except Exception:
        pass

    # Balance sheet
    debt_M = cash_M = None
    try:
        bs = t.quarterly_balance_sheet
        col = bs.columns[0]
        debt_M = _round((bs[col].get("Total Debt") or bs[col].get("Long Term Debt", 0)) / 1e6)
        cash_M = _round((bs[col].get("Cash And Cash Equivalents") or 0) / 1e6)
    except Exception:
        pass

    info = t.info
    return {
        "annual": {
            "revenue_M":        annual_rev,
            "gross_profit_M":   annual_gp,
            "operating_income_M": annual_op,
            "net_income_M":     annual_ni,
            "eps_diluted":      annual_eps,
        },
        "quarterly_recent": {
            "revenue_M":        q_rev,
            "gross_profit_M":   q_gp,
            "net_income_M":     q_ni,
            "eps_diluted":      q_eps,
        },
        "margins_pct": {
            "gross_margin":     gross_margin,
            "operating_margin": op_margin,
            "net_margin":       net_margin,
        },
        "balance_sheet": {
            "total_debt_M":     debt_M,
            "cash_M":           cash_M,
            "debt_to_equity":   _round(info.get("debtToEquity")),
        },
        "fcf_ttm_M":            fcf_M,
        "roe_pct":              _round((info.get("returnOnEquity") or 0) * 100),
        "ps_ratio":             _round(info.get("priceToSalesTrailing12Months")),
        "pb_ratio":             _round(info.get("priceToBook")),
        "dividend_yield_pct":   _round((info.get("dividendYield") or 0) * 100),
    }


def get_company_info(ticker: str) -> dict:
    """Return company profile: description, sector, industry, employees."""
    info = yf.Ticker(ticker).info
    return {
        "name":           info.get("longName"),
        "sector":         info.get("sector"),
        "industry":       info.get("industry"),
        "country":        info.get("country"),
        "employees":      info.get("fullTimeEmployees"),
        "market_cap_B":   _round((info.get("marketCap") or 0) / 1e9),
        "description":    (info.get("longBusinessSummary") or "")[:800],
        "website":        info.get("website"),
    }


def get_macro_indicators() -> dict:
    """Return current VIX, US 10Y yield, 2Y yield, DXY, and SPY 1-year return."""
    def _last(sym):
        try:
            h = yf.Ticker(sym).history(period="5d")
            return _round(h["Close"].dropna().iloc[-1]) if not h.empty else None
        except Exception:
            return None

    def _ret1y(sym):
        try:
            h = yf.Ticker(sym).history(period="1y")["Close"].dropna()
            return _round((h.iloc[-1] / h.iloc[0] - 1) * 100) if len(h) >= 2 else None
        except Exception:
            return None

    return {
        "vix":              _last("^VIX"),
        "us_10y_yield_pct": _last("^TNX"),
        "us_2y_yield_pct":  _last("^IRX"),
        "dxy":              _last("DX-Y.NYB"),
        "spy_1y_return_pct": _ret1y("SPY"),
        "as_of":            str(datetime.today().date()),
        "note": (
            "Yields from Yahoo Finance (^TNX=10Y, ^IRX=13-week proxy for 2Y). "
            "For PMI, GDP, Fed statements the AI should rely on its training knowledge "
            "and flag the recency limitation."
        ),
    }


# ---------------------------------------------------------------------------
# Tool dispatch
# ---------------------------------------------------------------------------

_TOOL_MAP = {
    "get_price_history":    lambda a: get_price_history(**a),
    "get_pe_history":       lambda a: get_pe_history(**a),
    "get_financials":       lambda a: get_financials(**a),
    "get_company_info":     lambda a: get_company_info(**a),
    "get_macro_indicators": lambda _: get_macro_indicators(),
}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_price_history",
            "description": (
                "Fetch historical stock price data: monthly close series, "
                "1m/3m/6m/1y/3y returns, 52-week high/low, beta, analyst target price."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Stock ticker symbol, e.g. AAPL"},
                    "years":  {"type": "integer", "description": "Years of history (default 3)"},
                },
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_pe_history",
            "description": (
                "Fetch trailing P/E history computed from annual + quarterly EPS. "
                "Returns current P/E, forward P/E, PEG, historical median/range, "
                "current percentile, and quarterly P/E readings."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string"},
                    "years":  {"type": "integer", "description": "Years of history (default 5)"},
                },
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_financials",
            "description": (
                "Fetch annual and recent quarterly income statement (revenue, EPS, gross/op/net income), "
                "margin trends, FCF, balance sheet (debt, cash, D/E), ROE, P/S, P/B, dividend yield."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string"},
                },
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_company_info",
            "description": (
                "Fetch company profile: full name, sector, industry, country, "
                "employee count, market cap, and business description."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string"},
                },
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_macro_indicators",
            "description": (
                "Fetch current macro indicators: VIX, US 10Y and 2Y Treasury yields, "
                "DXY (dollar index), and SPY 1-year return. "
                "Use this at the start of Layer 1 for macro context."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]


# ---------------------------------------------------------------------------
# System prompt lives in prompts.py


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

_PROVIDERS = {
    "openai":   ("https://api.openai.com/v1",                         "OPENAI_API_KEY",    "gpt-4o",             {}),
    "deepseek": ("https://api.deepseek.com/v1",                       "DEEPSEEK_API_KEY",  "deepseek-chat",      {}),
    "qwen":     ("https://dashscope.aliyuncs.com/compatible-mode/v1", "DASHSCOPE_API_KEY", "qwen-plus",          {}),
    "claude":   ("https://api.anthropic.com/v1",                      "ANTHROPIC_API_KEY", "claude-opus-4-7",  {"default_headers": {"anthropic-version": "2023-06-01"}}),
}


def run_skill(ticker: str, portfolio_context: str, provider: str, model_name: str | None):
    cfg = _PROVIDERS.get(provider)
    if cfg is None:
        sys.exit(f"Unknown provider '{provider}'. Choose from: {list(_PROVIDERS)}")

    base_url, env_key, default_model, client_kwargs = cfg
    api_key = os.getenv(env_key)
    if not api_key:
        sys.exit(f"Missing env var {env_key} for provider '{provider}'.")

    model = model_name or default_model
    client = OpenAI(base_url=base_url, api_key=api_key, **client_kwargs)

    user_msg = (
        f"Please conduct a complete five-layer analysis for **{ticker}**.\n\n"
        f"Today's date: {datetime.today().strftime('%Y-%m-%d')}\n"
    )
    if portfolio_context:
        user_msg += f"\nPortfolio context for Layer 5: {portfolio_context}\n"

    messages = [
        {"role": "system", "content": FIVE_LAYER_SYSTEM_PROMPT},
        {"role": "user",   "content": user_msg},
    ]

    print(f"\nStarting five-layer analysis for {ticker} via {provider.upper()} ({model})…\n")
    print("=" * 70)

    create_kwargs = dict(model=model, messages=messages, tools=TOOLS, tool_choice="auto")
    if provider != "claude":
        create_kwargs["temperature"] = 0.2

    call_count = 0
    while True:
        response = client.chat.completions.create(**create_kwargs)
        choice = response.choices[0]

        if choice.finish_reason == "tool_calls":
            messages.append(choice.message)
            for tc in choice.message.tool_calls:
                fn_name = tc.function.name
                fn_args = json.loads(tc.function.arguments)
                call_count += 1
                print(f"[Tool call {call_count}] {fn_name}({', '.join(f'{k}={v}' for k,v in fn_args.items())})")
                try:
                    result = _TOOL_MAP[fn_name](fn_args)
                except Exception as e:
                    result = {"error": str(e)}
                messages.append({
                    "role":         "tool",
                    "tool_call_id": tc.id,
                    "content":      json.dumps(result, default=str),
                })
        else:
            # Final text response
            content = choice.message.content
            print()
            print(content)
            print("\n" + "=" * 70)

            out_dir = os.path.join(os.path.dirname(__file__), "reports")
            os.makedirs(out_dir, exist_ok=True)
            timestamp = datetime.today().strftime("%Y%m%d_%H%M%S")
            out_path = os.path.join(out_dir, f"{ticker}_{timestamp}_{provider}.md")
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(f"# {ticker} — Five-Layer Analysis\n")
                f.write(f"**Date:** {datetime.today().strftime('%Y-%m-%d %H:%M:%S')}  \n")
                f.write(f"**Provider:** {provider} ({model})  \n\n")
                f.write("---\n\n")
                f.write(content)
            print(f"\nReport saved → {out_path}")
            break


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Five-layer AI stock research skill with tool calling.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("ticker", help="Stock ticker, e.g. AAPL")
    parser.add_argument(
        "--provider",
        default="qwen",
        choices=list(_PROVIDERS),
        help="AI provider (default: qwen)",
    )
    parser.add_argument("--model", default=None, help="Override default model for the provider")
    parser.add_argument(
        "--portfolio",
        default="",
        metavar="CONTEXT",
        help='Portfolio context for Layer 5, e.g. "AAPL 30%%, MSFT 20%%, cash 50%%"',
    )
    args = parser.parse_args()

    run_skill(
        ticker=args.ticker.upper(),
        portfolio_context=args.portfolio,
        provider=args.provider,
        model_name=args.model,
    )


if __name__ == "__main__":
    main()
