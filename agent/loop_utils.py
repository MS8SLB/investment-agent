"""
Utilities for controlling token cost in agentic loops.
- prune_messages: drop old turns to prevent context blowup
- truncate_tool_result: cap large tool outputs before appending to messages
"""

from __future__ import annotations

# Tools whose results must never be truncated (already compressed upstream)
_NO_TRUNCATE_TOOLS: frozenset = frozenset({
    "research_stocks_parallel",
    "challenge_buy_theses",
})

# Tools that return large blobs we aggressively cap
_AGGRESSIVE_TRUNCATE_TOOLS: frozenset = frozenset({
    "get_stock_news",
    "get_analyst_upgrades",
    "get_insider_activity",
    "screen_stocks",
    "get_stock_universe",
    "get_international_universe",
    "get_trade_outcomes",
    "check_news_alerts",
    "get_triaged_alerts",
    "run_backtest",
    "get_sector_exposure",
    "get_macro_environment",
    "get_market_summary",
})


def prune_messages(messages: list, max_turns: int = 10) -> list:
    """
    Keep messages[0] (initial user prompt) + the last max_turns*2 messages.
    Prevents the context window from growing unboundedly across tool calls.
    """
    if not messages:
        return messages
    threshold = 1 + max_turns * 2
    if len(messages) <= threshold:
        return messages
    return messages[:1] + messages[-(max_turns * 2):]


def truncate_tool_result(tool_name: str, content: str, max_chars: int = 4000) -> str:
    """
    Truncate tool result content to max_chars before appending to messages.
    - Never truncates _NO_TRUNCATE_TOOLS
    - Uses a tighter limit for _AGGRESSIVE_TRUNCATE_TOOLS (max_chars // 2)
    - For JSON content, tries to trim arrays before falling back to raw truncation
    """
    if tool_name in _NO_TRUNCATE_TOOLS:
        return content

    limit = (max_chars // 2) if tool_name in _AGGRESSIVE_TRUNCATE_TOOLS else max_chars

    if len(content) <= limit:
        return content

    # Try JSON-aware trimming first
    try:
        import json
        data = json.loads(content)
        if isinstance(data, list) and len(data) > 3:
            trimmed = data[:3]
            trimmed.append({"_truncated": f"... {len(data) - 3} more items omitted"})
            result = json.dumps(trimmed)
            if len(result) <= limit:
                return result
        elif isinstance(data, dict):
            # Drop large string values
            cleaned = {}
            for k, v in data.items():
                if isinstance(v, str) and len(v) > 500:
                    cleaned[k] = v[:500] + "... [truncated]"
                elif isinstance(v, list) and len(v) > 5:
                    cleaned[k] = v[:5] + [f"... {len(v)-5} more"]
                else:
                    cleaned[k] = v
            result = json.dumps(cleaned)
            if len(result) <= limit:
                return result
    except (json.JSONDecodeError, TypeError):
        pass

    # Raw truncation fallback
    return content[:limit] + f"\n... [truncated — {len(content) - limit} chars omitted]"
