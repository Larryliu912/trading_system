import os
from openai import OpenAI

def get_ai_market_signal(json_payload, provider="openai", model_name=None, timeframe="15m"):
    """
    Sends the structured market data payload to the specified AI provider 
    using an OpenAI-compatible client configuration.
    
    Supported providers: 'openai', 'deepseek', 'qwen'
    """
    
    # 1. Dynamically configure the base URL and API key based on the provider
    if provider == "openai":
        base_url = "https://api.openai.com/v1"
        api_key = os.getenv("OPENAI_API_KEY")
        default_model = "gpt-4o"
        
    elif provider == "deepseek":
        base_url = "https://api.deepseek.com/v1"
        api_key = os.getenv("DEEPSEEK_API_KEY")
        default_model = "deepseek-chat" # DeepSeek-V3 / DeepSeek-R1
        
    elif provider == "qwen":
        # Alibaba DashScope OpenAI-compatible endpoint
        base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        api_key = os.getenv("DASHSCOPE_API_KEY")
        default_model = "qwen-plus" # e.g., qwen-max, qwen-plus, or qwen-turbo
        
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    if not api_key:
        raise ValueError(f"Missing API key environment variable for {provider.upper()}.")

    # 2. Initialize the client with the custom base_url
    client = OpenAI(base_url=base_url, api_key=api_key)
    
    # Use the user-provided model name or fall back to the provider default
    selected_model = model_name if model_name else default_model

    # 3. Define the system instructions for market analysis
    if timeframe == "day":
        system_prompt = (
            "You are an expert quantitative trading system assistant specializing in "
            "Daily Timeframe Analysis for Equity and Volatility Futures.\n\n"
            "Your task is to analyze the provided market JSON payload containing "
            "daily data for S&P 500 futures and VIX, then predict the NEXT TRADING DAY's direction.\n"
            "Respond strictly in the following format:\n"
            "### BIAS ASSESSMENT\n[Determine if the structural daily trend and momentum align for tomorrow]\n\n"
            "### VOLATILITY CHECK\n[Analyze VIX behavior and what it signals for next day's risk environment]\n\n"
            "### STRATEGIC SIGNAL\n[BUY / SELL / HOLD with a concise 1-sentence justification for next day's direction]\n\n"
            "### Confidence\n[A percentage confidence level for the signal, e.g., 85%]"
        )
    elif timeframe == "4h":
        system_prompt = (
            "You are an expert quantitative trading system assistant specializing in "
            "Multi-Timeframe Swing Analysis (MTA) for Equity and Volatility Futures.\n\n"
            "Your task is to analyze the provided market JSON payload containing "
            "Macro (1d) and Swing (4h) data for S&P 500 and VIX futures, then predict the NEXT 4-HOUR bar's direction.\n"
            "Respond strictly in the following format:\n"
            "### BIAS ASSESSMENT\n[Determine if the daily structural trend and 4h swing momentum align]\n\n"
            "### VOLATILITY CHECK\n[Analyze VIX behavior relative to the 4h price action]\n\n"
            "### STRATEGIC SIGNAL\n[BUY / SELL / HOLD with a concise 1-sentence justification for the next 4 hours]\n\n"
            "### Confidence\n[A percentage confidence level for the signal, e.g., 85%]"
        )
    else:
        system_prompt = (
            "You are an expert quantitative trading system assistant specializing in "
            "Three-Timeframe Analysis (MTA) for Equity and Volatility Futures.\n\n"
            "Your task is to analyze the provided market JSON payload containing "
            "Macro (1d), Swing (4h), and Micro (15m) data for S&P 500 and VIX futures, "
            "then predict the NEXT 15-MINUTE bar's direction.\n"
            "Respond strictly in the following format:\n"
            "### BIAS ASSESSMENT\n[Assess alignment across all three timeframes: daily trend, 4h swing, and 15m momentum]\n\n"
            "### VOLATILITY CHECK\n[Analyze VIX behavior relative to the 4h swing and 15m price action]\n\n"
            "### STRATEGIC SIGNAL\n[BUY / SELL / HOLD with a concise 1-sentence tactical justification]\n\n"
            "### Confidence\n[A percentage confidence level for the signal, e.g., 85%]"
        )

    # 4. Execute the API call
    print(f"Sending market payload to {provider.upper()} ({selected_model})...")
    
    response = client.chat.completions.create(
        model=selected_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Analyze this current market data asset state:\n{json_payload}"}
        ],
        temperature=0.2 # Lower temperature for more consistent, analytical logic
    )
    
    return response.choices[0].message.content


# =====================================================================
# Example Usage:
# (Make sure to set your environment variables before running, e.g., export DEEPSEEK_API_KEY="your_key")
# =====================================================================
if __name__ == "__main__":
    # Assuming 'ai_ready_json' is the variable generated from your previous data preparation script
    try:
        # Example 1: Standard OpenAI Call
        # analysis_openai = get_ai_market_signal(ai_ready_json, provider="openai")
        # print(analysis_openai)
        
        # Example 2: Switching to DeepSeek with the exact same payload
        # analysis_deepseek = get_ai_market_signal(ai_ready_json, provider="deepseek")
        # print(analysis_deepseek)
        
        # Example 3: Switching to Qwen (Alibaba)
        # analysis_qwen = get_ai_market_signal(ai_ready_json, provider="qwen")
        # print(analysis_qwen)
        
        print("Connector configured successfully! Uncomment an example line above to execute live requests.")
        
    except NameError:
        print("Setup complete. To run live, pass your generated 'ai_ready_json' string into get_ai_market_signal().")