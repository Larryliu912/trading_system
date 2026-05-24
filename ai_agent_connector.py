import os
from openai import OpenAI


_ANALYSIS_CONFIGS = {
    "pure-spy": {
        "identity": "S&P 500 Futures specialist using price action and technical indicators only",
        "data_desc": "S&P 500 futures price data",
        "predict_target": "S&P 500 Futures",
    },
    "pure-vix": {
        "identity": "VIX volatility specialist using VIX data only",
        "data_desc": "VIX volatility index data",
        "predict_target": "VIX",
    },
    "hybrid-spy": {
        "identity": "quantitative trading system assistant specializing in Equity and Volatility Futures",
        "data_desc": "S&P 500 futures and VIX data",
        "predict_target": "S&P 500 Futures",
    },
    "hybrid-vix": {
        "identity": "volatility trading specialist analyzing S&P 500 and VIX correlations",
        "data_desc": "S&P 500 futures and VIX data",
        "predict_target": "VIX",
    },
}

_TIMEFRAME_CONFIGS = {
    "day": {
        "scope": "Daily Timeframe Analysis",
        "horizon": "NEXT TRADING DAY",
        "bar_label": "next trading day's",
        "bias_note": "Determine if the structural daily trend and momentum align",
    },
    "4h": {
        "scope": "Multi-Timeframe Swing Analysis (1d + 4h)",
        "horizon": "NEXT 4-HOUR BAR",
        "bar_label": "next 4-hour bar's",
        "bias_note": "Determine if the daily structural trend and 4h swing momentum align",
    },
    "15m": {
        "scope": "Three-Timeframe Analysis (1d + 4h + 15m)",
        "horizon": "NEXT 15-MINUTE BAR",
        "bar_label": "next 15-minute bar's",
        "bias_note": "Assess alignment across all three timeframes: daily trend, 4h swing, and 15m momentum",
    },
}


def get_ai_market_signal(json_payload, provider="qwen", model_name=None, timeframe="day", analysis="hybrid-spy"):
    """
    Sends the structured market data payload to the specified AI provider
    using an OpenAI-compatible client configuration.

    Supported providers: 'openai', 'deepseek', 'qwen'
    Analysis modes: 'pure-spy', 'pure-vix', 'hybrid-spy', 'hybrid-vix'
    """

    if provider == "openai":
        base_url = "https://api.openai.com/v1"
        api_key = os.getenv("OPENAI_API_KEY")
        default_model = "gpt-4o"
    elif provider == "deepseek":
        base_url = "https://api.deepseek.com/v1"
        api_key = os.getenv("DEEPSEEK_API_KEY")
        default_model = "deepseek-v4-pro"
    elif provider == "qwen":
        base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        api_key = os.getenv("DASHSCOPE_API_KEY")
        default_model = "qwen-plus"
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    if not api_key:
        raise ValueError(f"Missing API key environment variable for {provider.upper()}.")

    client = OpenAI(base_url=base_url, api_key=api_key)
    selected_model = model_name if model_name else default_model

    acfg = _ANALYSIS_CONFIGS.get(analysis, _ANALYSIS_CONFIGS["hybrid-spy"])
    tfcfg = _TIMEFRAME_CONFIGS.get(timeframe, _TIMEFRAME_CONFIGS["15m"])
    target = acfg["predict_target"]

    system_prompt = (
        f"You are an expert {acfg['identity']} specializing in {tfcfg['scope']}.\n\n"
        f"Your task is to analyze the provided market JSON payload containing {acfg['data_desc']}, "
        f"then predict the {tfcfg['horizon']}'s direction for {target}. You can also calculate the any technical indicators needed by yourself.\n"
        "Respond strictly in the following format:\n"
        f"### BIAS ASSESSMENT\n[{tfcfg['bias_note']} for {target}]\n\n"
        f"### VOLATILITY CHECK\n[Analyze the volatility context and risk environment relevant to {target}]\n\n"
        f"### STRATEGIC SIGNAL\n[BUY / SELL / HOLD {target} with a concise 1-sentence justification "
        f"for the {tfcfg['bar_label']} direction]\n\n"
        "### Confidence\n[A percentage confidence level for the signal, e.g., 85%]"
    )

    print(f"Sending market payload to {provider.upper()} ({selected_model})...")

    response = client.chat.completions.create(
        model=selected_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Analyze this current market data asset state:\n{json_payload}"}
        ],
        temperature=0.2
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    try:
        print("Connector configured successfully! Uncomment an example line above to execute live requests.")
    except NameError:
        print("Setup complete. To run live, pass your generated 'ai_ready_json' string into get_ai_market_signal().")
