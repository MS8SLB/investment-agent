"""
Utilities for controlling token cost in agentic loops.
- prune_messages: drop old turns to prevent context blowup, with optional
  summarization of dropped content so decisions aren't silently lost
- truncate_tool_result: cap large tool outputs before appending to messages
"""

from __future__ import annotations

import json
import os

# Tools whose results must never be truncated (already compressed upstream)
_NO_TRUNCATE_TOOLS: frozenset = frozenset({
    "research_stocks_parallel",
    "challenge_buy_theses",
    "screen_stocks",          # handler already caps at top-25; truncation cuts to 5
})

# Tools that return large blobs we aggressively cap
_AGGRESSIVE_TRUNCATE_TOOLS: frozenset = frozenset({
    "get_stock_news",
    "get_analyst_upgrades",
    "get_insider_activity",
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


def add_cache_control(tools: list) -> list:
    """
    Mark the last tool definition with cache_control so the entire tool list
    is treated as a single cacheable prefix by the Anthropic API.
    Returns a shallow copy — the original list is not modified.
    """
    if not tools:
        return tools
    result = list(tools)
    last = dict(result[-1])
    last["cache_control"] = {"type": "ephemeral"}
    result[-1] = last
    return result


def _summarize_dropped_turns(dropped: list) -> str:
    """
    Produce a compact bullet-point summary of dropped conversation turns so that
    decisions made in pruned history aren't silently lost.  Uses a lightweight
    Claude Haiku call; falls back to a plain text extraction if the API is
    unavailable.
    """
    # Extract human-readable fragments from the dropped turns
    fragments: list[str] = []
    for msg in dropped:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if isinstance(content, str):
            fragments.append(f"[{role}] {content[:300]}")
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        fragments.append(f"[{role}] {block['text'][:300]}")
                    elif block.get("type") == "tool_use":
                        fragments.append(f"[tool_call] {block['name']}({json.dumps(block.get('input', {}))[:150]})")
                    elif block.get("type") == "tool_result":
                        fragments.append(f"[tool_result] {str(block.get('content', ''))[:200]}")
    if not fragments:
        return ""

    raw = "\n".join(fragments)

    try:
        import anthropic as _anthropic
        _client = _anthropic.Anthropic()
        resp = _client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=512,
            messages=[{
                "role": "user",
                "content": (
                    "You are a note-taker for an investment agent session. "
                    "Summarise the following conversation fragments in ≤10 bullet points. "
                    "Focus on: investment decisions made, tickers researched or watchlisted, "
                    "macro observations, and any rules-violation flags. "
                    "Be terse — each bullet max 20 words.\n\n"
                    f"{raw[:3000]}"
                ),
            }],
        )
        text = resp.content[0].text if resp.content else ""
        return text.strip()
    except Exception:
        # Fallback: plain truncated concat
        return "Prior context (summarization unavailable):\n" + raw[:1000]


def prune_messages(messages: list, max_turns: int = 20) -> list:
    """
    Keep messages[0] (initial user prompt) + the last max_turns*2 messages.
    Dropped turns are summarized via a lightweight Haiku call and injected back
    as a system note so decisions from pruned history aren't silently lost.
    """
    if not messages:
        return messages
    threshold = 1 + max_turns * 2
    if len(messages) <= threshold:
        return messages

    kept_tail = messages[-(max_turns * 2):]
    dropped = messages[1:len(messages) - (max_turns * 2)]  # everything between [0] and tail

    summary = _summarize_dropped_turns(dropped)

    retained = list(messages[:1])  # initial user prompt
    if summary:
        retained.append({
            "role": "user",
            "content": f"[Session memory — earlier turns were pruned to manage context. Key decisions so far:]\n{summary}",
        })
        retained.append({
            "role": "assistant",
            "content": "Understood. I have noted the prior session decisions and will continue from there.",
        })
    retained.extend(kept_tail)
    return retained


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
