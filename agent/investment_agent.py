"""
Claude-powered investment agent.
Uses an agentic loop with tool use to research and manage a paper portfolio.
"""

import json
import os
from typing import Optional

import anthropic

from agent.tools import TOOL_DEFINITIONS, handle_tool_call
from agent import portfolio


SYSTEM_PROMPT = """You are an expert long-term investment portfolio manager running a paper trading portfolio.

## Your Investment Philosophy
- **Long-term focus**: You invest with a 3-10 year horizon. You avoid panic selling and short-term noise.
- **Fundamental analysis**: You evaluate companies based on earnings growth, profit margins, ROE, P/E ratio, competitive moats, and balance sheet health.
- **Diversification**: You spread investments across sectors (tech, healthcare, consumer, financials, energy, etc.) to reduce risk.
- **Quality over quantity**: You prefer high-quality businesses with durable competitive advantages over speculative plays.
- **Valuation discipline**: You avoid overpaying. You look for companies trading at reasonable valuations relative to their growth prospects.
- **Dividend consideration**: For stability, you appreciate companies with consistent dividend growth.

## Decision Framework
When considering a buy:
1. Check the current portfolio status to understand available cash and existing positions
2. Research the stock's fundamentals (P/E, margins, ROE, growth)
3. Check price history to understand valuation context
4. Consider sector exposure — avoid over-concentrating in one sector
5. Make the purchase if conviction is high

When considering a sell:
1. Fundamentals have deteriorated (not just price drop)
2. Better opportunity exists and capital reallocation makes sense
3. Position has grown too large (>20% of portfolio)

## Portfolio Rules
- Maximum position size: 20% of total portfolio value
- Target: 8-15 stocks for good diversification
- Keep at least 10% in cash as a buffer
- Prefer established companies with track records, but thoughtful allocation to high-growth names is acceptable

## Today's Context
Today's date is February 19, 2026. You are managing a paper trading portfolio starting with $100,000.

## Communication Style
- Be analytical and data-driven in your decisions
- Explain your reasoning clearly
- After taking actions, summarize what you did and why
- Be honest about uncertainty and risks

Always use your tools to gather real data before making investment decisions. Never guess at prices."""


def run_agent_session(
    user_prompt: str,
    model: Optional[str] = None,
    max_iterations: int = 20,
    on_text: Optional[callable] = None,
    on_tool_call: Optional[callable] = None,
    on_tool_result: Optional[callable] = None,
) -> str:
    """
    Run one agent session with the given user prompt.

    Args:
        user_prompt: The user's instruction or question.
        model: Claude model ID (defaults to env var CLAUDE_MODEL or claude-opus-4-6).
        max_iterations: Safety cap on tool-use iterations.
        on_text: Callback(text: str) for streaming text chunks.
        on_tool_call: Callback(tool_name: str, tool_input: dict).
        on_tool_result: Callback(tool_name: str, result: any).

    Returns:
        The final text response from the agent.
    """
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    model = model or os.environ.get("CLAUDE_MODEL", "claude-opus-4-6")

    messages = [{"role": "user", "content": user_prompt}]
    final_text = ""

    for iteration in range(max_iterations):
        response = client.messages.create(
            model=model,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=TOOL_DEFINITIONS,
            messages=messages,
        )

        # Collect text and tool use blocks
        text_parts = []
        tool_calls = []

        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
                if on_text:
                    on_text(block.text)
            elif block.type == "tool_use":
                tool_calls.append(block)
                if on_tool_call:
                    on_tool_call(block.name, block.input)

        if text_parts:
            final_text = "\n".join(text_parts)

        # If no tool calls, the agent is done
        if response.stop_reason == "end_turn" or not tool_calls:
            break

        # Append assistant message
        messages.append({"role": "assistant", "content": response.content})

        # Execute tool calls and build tool results
        tool_results = []
        for tc in tool_calls:
            result = handle_tool_call(tc.name, tc.input)
            if on_tool_result:
                on_tool_result(tc.name, result)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tc.id,
                "content": json.dumps(result, default=str),
            })

        messages.append({"role": "user", "content": tool_results})

    portfolio.log_agent_message(f"Session completed: {user_prompt[:100]}")
    return final_text


def run_portfolio_review(model: Optional[str] = None, **kwargs) -> str:
    """Run a full autonomous portfolio review and rebalancing session."""
    prompt = """
Please conduct a comprehensive portfolio review and take appropriate investment actions:

1. Start by checking the current portfolio status
2. Check overall market conditions
3. Review any existing positions for health
4. Research 2-3 potential new investment opportunities across different sectors
5. Make buy/sell decisions based on your analysis
6. Provide a summary of actions taken and the portfolio's current state

Focus on building a diversified, high-quality long-term portfolio.
"""
    return run_agent_session(prompt, model=model, **kwargs)


def run_custom_prompt(prompt: str, model: Optional[str] = None, **kwargs) -> str:
    """Run the agent with a custom user prompt."""
    return run_agent_session(prompt, model=model, **kwargs)
