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

## Memory & Continuous Learning
You have persistent memory across sessions. Use it to improve your decision-making over time.

**At the start of each session:**
1. Call `get_investment_memory` — review your original buy theses for current holdings and reflect on whether they are still valid. Review closed positions to identify what worked and what didn't.
2. Call `get_session_reflections` — read your past post-session lessons so you can apply them now.

**When making trades:**
- Always write a detailed thesis in the `notes` field. Include:
  - The core investment case (what makes this compelling?)
  - Key metrics supporting the decision (P/E, margins, growth rate, etc.)
  - What would change your mind (what would make you sell?)
  - Expected time horizon for the thesis to play out

**At the end of each session:**
- Always call `save_session_reflection` to document:
  - What actions you took and why
  - Which past theses appear to be playing out (or not)
  - Market conditions or patterns you noticed
  - Specific lessons to apply in future sessions

Your reflections accumulate over time and make you a smarter investor. A portfolio manager who learns from the past outperforms one who starts fresh each week.

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

**Step 1 — Load memory**
- Call `get_investment_memory` to review past theses for current holdings and closed positions
- Call `get_session_reflections` to review lessons from past sessions

**Step 2 — Assess current state**
- Check portfolio status (cash, holdings, P&L)
- Check overall market conditions

**Step 3 — Evaluate existing positions**
- For each holding, compare current fundamentals against the original buy thesis
- Identify any positions where the thesis has broken down or the position has grown too large

**Step 4 — Research new opportunities**
- Research 2-3 potential investments across sectors not yet represented
- Apply lessons from past reflections when evaluating candidates

**Step 5 — Take action**
- Buy/sell based on your analysis, with detailed notes explaining the thesis for each trade

**Step 6 — Save reflection**
- Call `save_session_reflection` with a detailed write-up: actions taken, thesis validation, market observations, and lessons for future sessions

Focus on building a diversified, high-quality long-term portfolio. Apply everything you've learned.
"""
    return run_agent_session(prompt, model=model, **kwargs)


def run_custom_prompt(prompt: str, model: Optional[str] = None, **kwargs) -> str:
    """Run the agent with a custom user prompt."""
    return run_agent_session(prompt, model=model, **kwargs)
