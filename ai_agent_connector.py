"""
AI Market Signal Connector
Provides get_ai_market_signal() used by common/data_postpreposs.py.
"""
import os

from openai import OpenAI

_PROVIDERS = {
    "openai":   ("https://api.openai.com/v1",                         "OPENAI_API_KEY",    "gpt-4o",          {}),
    "deepseek": ("https://api.deepseek.com/v1",                       "DEEPSEEK_API_KEY",  "deepseek-v4-pro", {}),
    "qwen":     ("https://dashscope.aliyuncs.com/compatible-mode/v1", "DASHSCOPE_API_KEY", "qwen-plus",       {}),
    "claude":   ("https://api.anthropic.com/v1",                      "ANTHROPIC_API_KEY", "claude-opus-4-7",
                 {"default_headers": {"anthropic-version": "2023-06-01"}}),
}

_SYSTEM_PROMPT = """You are a quantitative trading signal generator for S&P 500 and VIX futures.
You receive multi-timeframe market data in JSON format. Analyze the technical indicators and output a single directional signal.

Your response MUST follow this exact format:

### Signal
[BUY / SELL / HOLD]

### Reasoning
[2-3 sentences covering the key indicators driving your decision]

### Confidence
[percentage, e.g., 65%]

Rules:
- BUY: expect the target asset price to rise in the next bar/period
- SELL: expect the target asset price to fall in the next bar/period
- HOLD: no clear edge — stay flat
- Be realistic about confidence (typical range 45-70%)
"""


def get_ai_market_signal(
    ai_payload: str,
    provider: str = "openai",
    model_name: str | None = None,
    timeframe: str = "15m",
    analysis: str = "hybrid-spy",
) -> str:
    cfg = _PROVIDERS.get(provider)
    if cfg is None:
        raise ValueError(f"Unknown provider '{provider}'. Choose from: {list(_PROVIDERS)}")

    base_url, env_key, default_model, client_kwargs = cfg
    api_key = os.getenv(env_key)
    if not api_key:
        raise EnvironmentError(f"Missing env var {env_key} for provider '{provider}'.")

    model = model_name or default_model
    client = OpenAI(base_url=base_url, api_key=api_key, **client_kwargs)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": ai_payload},
        ],
        max_tokens=300,
        temperature=0.3,
    )
    return response.choices[0].message.content or "HOLD"
