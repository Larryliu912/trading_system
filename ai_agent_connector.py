"""
AI Market Signal Connector
Provides get_ai_market_signal() used by common/data_postpreposs.py.
"""
import os

from openai import OpenAI

from prompts import MARKET_SIGNAL_SYSTEM_PROMPT

_PROVIDERS = {
    "openai":   ("https://api.openai.com/v1",                         "OPENAI_API_KEY",    "gpt-4o",          {}),
    "deepseek": ("https://api.deepseek.com/v1",                       "DEEPSEEK_API_KEY",  "deepseek-v4-pro", {}),
    "qwen":     ("https://dashscope.aliyuncs.com/compatible-mode/v1", "DASHSCOPE_API_KEY", "qwen-plus",       {}),
    "claude":   ("https://api.anthropic.com/v1",                      "ANTHROPIC_API_KEY", "claude-opus-4-7",
                 {"default_headers": {"anthropic-version": "2023-06-01"}}),
}


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
            {"role": "system", "content": MARKET_SIGNAL_SYSTEM_PROMPT},
            {"role": "user", "content": ai_payload},
        ],
        max_tokens=300,
        temperature=0.3,
    )
    return response.choices[0].message.content or "HOLD"
