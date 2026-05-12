#!/usr/bin/env python3
"""
Watchlist & Portfolio Monitor

Prints a live snapshot of:
  - Watchlist: current price vs target, % away from entry
  - Holdings: current price, gain/loss, % return
  - Shadow portfolio: how passed stocks have moved since the pass decision

Usage:
    python monitor.py                 # one-shot snapshot
    python monitor.py --watch         # refresh every 60s (Ctrl-C to stop)
    python monitor.py --watch --interval 300   # refresh every 5 min
    python monitor.py --alert         # lightweight alert check (8-Ks + price triggers)
    python monitor.py --alert --since 2025-01-01  # alerts since a specific date
"""

import argparse
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent))

import warnings
warnings.filterwarnings("ignore", category=Warning, module="urllib3")

from agent import market_data, portfolio
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

console = Console()

DIVIDER = "━" * 60


def _price(ticker: str):
    q = market_data.get_stock_quote(ticker)
    return q.get("price")


def _pct(current: float, reference: float) -> float:
    return (current - reference) / reference * 100


def _color(val: float) -> str:
    if val > 0:
        return "green"
    if val < 0:
        return "red"
    return "white"


def _fmt_pct(val: float) -> str:
    sign = "+" if val >= 0 else ""
    return f"{sign}{val:.1f}%"


def render(console: Console) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    console.print(f"\n[bold]{DIVIDER}[/bold]")
    console.print(f"[bold]WATCHLIST MONITOR  {now}[/bold]")
    console.print(f"[bold]{DIVIDER}[/bold]\n")

    # ── Watchlist ──────────────────────────────────────────────────────────────
    watchlist = portfolio.get_watchlist()
    active_list  = [w for w in watchlist if w.get("tier", "active") == "active"]
    monitor_list = [w for w in watchlist if w.get("tier") == "monitor"]
    no_target    = [w for w in active_list if w["target_entry_price"] is None]
    actionable   = [w for w in active_list if w["target_entry_price"] is not None]

    def _watchlist_table(items, label, dim_style=False):
        if not items:
            console.print(f"[bold]{label}[/bold]  [dim](none)[/dim]\n")
            return
        tbl = Table(box=box.SIMPLE_HEAVY, show_edge=False, pad_edge=False,
                    expand=False, min_width=60)
        tbl.add_column("Ticker", style="bold dim" if dim_style else "bold", min_width=6, no_wrap=True)
        tbl.add_column("Price",  justify="right", min_width=9, no_wrap=True)
        tbl.add_column("Target", justify="right", min_width=8, no_wrap=True)
        tbl.add_column("Away",   justify="right", min_width=7, no_wrap=True)
        tbl.add_column("Status", min_width=14, no_wrap=True)
        tbl.add_column("Note", style="dim", no_wrap=True, max_width=35)

        for w in items:
            ticker = w["ticker"]
            target = w["target_entry_price"]
            price  = _price(ticker)
            note   = (w["reason"] or "").split("—")[0].strip()[:40]

            if price is None or not target:
                tbl.add_row(ticker, "[dim]n/a[/dim]", f"${target:,.0f}" if target else "—", "—", "—", note)
                continue

            away_pct = _pct(price, target)
            color    = _color(-away_pct)

            if away_pct <= 0:
                status = "[bold green]▶ AT TARGET[/bold green]"
            elif away_pct <= 5:
                status = "[yellow]⚡ NEAR TARGET[/yellow]"
            elif away_pct <= 30:
                status = "[white]watching[/white]"
            else:
                status = "[dim]above target[/dim]"

            tbl.add_row(
                ticker,
                f"${price:,.2f}",
                f"${target:,.0f}",
                f"[{color}]{_fmt_pct(away_pct)}[/{color}]",
                status,
                note,
            )
        console.print(f"[bold]{label}[/bold]")
        console.print(tbl)

    _watchlist_table(actionable, "WATCHLIST — ACTIVE")
    if no_target:
        console.print("[dim]  Awaiting target price:[/dim] " +
                      ", ".join(w["ticker"] for w in no_target) + "\n")
    _watchlist_table(monitor_list, "WATCHLIST — MONITOR (thesis kept, price too far)", dim_style=True)

    # ── Holdings ───────────────────────────────────────────────────────────────
    holdings = portfolio.get_holdings()
    if holdings:
        console.print("[bold]HOLDINGS[/bold]")
        htbl = Table(box=box.SIMPLE_HEAVY, show_edge=False, pad_edge=False,
                     expand=False, min_width=60)
        htbl.add_column("Ticker",    style="bold", min_width=6, no_wrap=True)
        htbl.add_column("Shares",    justify="right", min_width=8, no_wrap=True)
        htbl.add_column("Avg Cost",  justify="right", min_width=9, no_wrap=True)
        htbl.add_column("Price",     justify="right", min_width=9, no_wrap=True)
        htbl.add_column("Return",    justify="right", min_width=8, no_wrap=True)
        htbl.add_column("Gain/Loss", justify="right", min_width=10, no_wrap=True)

        total_cost = total_value = 0.0
        for h in holdings:
            ticker   = h["ticker"]
            shares   = h["shares"]
            avg_cost = h["avg_cost"]
            price    = _price(ticker)

            cost_basis = shares * avg_cost
            total_cost += cost_basis

            if price is not None:
                mkt_val   = shares * price
                total_value += mkt_val
                ret       = _pct(price, avg_cost)
                gain      = mkt_val - cost_basis
                col       = _color(ret)
                htbl.add_row(
                    ticker,
                    f"{shares:,.2f}",
                    f"${avg_cost:,.2f}",
                    f"${price:,.2f}",
                    f"[{col}]{_fmt_pct(ret)}[/{col}]",
                    f"[{col}]${gain:+,.0f}[/{col}]",
                )
            else:
                total_value += cost_basis
                htbl.add_row(ticker, f"{shares:,.2f}", f"${avg_cost:,.2f}",
                             "[dim]n/a[/dim]", "—", "—")

        console.print(htbl)
        total_ret = _pct(total_value, total_cost) if total_cost else 0
        col = _color(total_ret)
        console.print(
            f"  Portfolio equity: [bold]${total_value:,.0f}[/bold]  "
            f"Total return: [{col}][bold]{_fmt_pct(total_ret)}[/bold][/{col}]\n"
        )
    else:
        console.print("[bold]HOLDINGS[/bold]  [dim](none)[/dim]\n")

    # ── Shadow portfolio ───────────────────────────────────────────────────────
    shadow = portfolio.get_shadow_positions()
    if shadow:
        console.print("[bold]SHADOW (passed stocks)[/bold]")
        stbl = Table(box=box.SIMPLE_HEAVY, show_edge=False, pad_edge=False,
                     expand=False, min_width=60)
        stbl.add_column("Ticker",   style="bold", min_width=6, no_wrap=True)
        stbl.add_column("Passed @", justify="right", min_width=9, no_wrap=True)
        stbl.add_column("Now",      justify="right", min_width=9, no_wrap=True)
        stbl.add_column("Move",     justify="right", min_width=7, no_wrap=True)
        stbl.add_column("Decision", min_width=12, no_wrap=True)
        stbl.add_column("Reason", style="dim", no_wrap=True, max_width=35)

        for s in shadow:
            ticker = s["ticker"]
            passed_price = s["price_at_consideration"]
            price = _price(ticker)

            if price is not None and passed_price:
                move = _pct(price, passed_price)
                col  = _color(-move)   # green = stock fell → pass was correct
                decision = "[green]✓ validated[/green]" if move <= 0 else "[red]✗ missed[/red]"
                move_str = f"[{col}]{_fmt_pct(move)}[/{col}]"
                price_str = f"${price:,.2f}"
            else:
                move_str = "—"
                price_str = "[dim]n/a[/dim]"
                decision = "—"

            short_reason = (s["reason_passed"] or "").split(";")[0].strip()[:40]
            stbl.add_row(
                ticker,
                f"${passed_price:,.2f}" if passed_price else "—",
                price_str,
                move_str,
                decision,
                short_reason,
            )

        console.print(stbl)


def check_alerts(since: Optional[str] = None) -> int:
    """
    Lightweight alert check — no agent session, just data checks.

    Checks:
    1. New 8-K filings for held positions since `since` date (or last 7 days)
    2. Watchlist items at or below target entry price (buy triggers)
    3. Holdings with a price move > 10% since last close

    Returns: number of actionable alerts found.
    """
    from agent import sec_data

    if since is None:
        since_dt = datetime.now() - timedelta(days=7)
        since = since_dt.strftime("%Y-%m-%d")

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    console.print(f"\n[bold]{DIVIDER}[/bold]")
    console.print(f"[bold]ALERT CHECK  {now_str}  (since {since})[/bold]")
    console.print(f"[bold]{DIVIDER}[/bold]\n")

    alert_count = 0

    # ── 1. 8-K material event alerts for holdings ──────────────────────────────
    holdings = portfolio.get_holdings()
    if holdings:
        console.print("[bold]8-K MATERIAL EVENTS (held positions)[/bold]")
        any_events = False
        for h in holdings:
            ticker = h["ticker"]
            try:
                since_days = max(1, (datetime.now() - datetime.strptime(since, "%Y-%m-%d")).days)
                events = sec_data.get_material_events(ticker, days=since_days)
                if isinstance(events, dict) and events.get("error"):
                    continue
                filings = []
                if isinstance(events, dict):
                    filings = events.get("events", events.get("filings", []))
                elif isinstance(events, list):
                    filings = events
                if filings:
                    any_events = True
                    alert_count += len(filings)
                    for f in filings[:3]:
                        item = f.get("item", f.get("items", ""))
                        desc = f.get("description", f.get("form", "8-K"))
                        date = f.get("date", f.get("filed", ""))[:10]
                        severity = "[bold red]⚠[/bold red]" if any(
                            k in str(item) for k in ("5.02", "4.01", "4.02", "1.03", "2.06")
                        ) else "[yellow]•[/yellow]"
                        console.print(f"  {severity} [bold]{ticker}[/bold]  {date}  Item {item}: {desc[:80]}")
            except Exception:
                continue
        if not any_events:
            console.print(f"  [dim]No material 8-K filings found since {since}[/dim]")
        console.print()

    # ── 2. Watchlist price triggers ───────────────────────────────────────────
    watchlist = portfolio.get_watchlist()
    actionable = [w for w in watchlist if w.get("target_entry_price") and w.get("tier", "active") == "active"]

    if actionable:
        triggered = []
        approaching = []
        for w in actionable:
            ticker = w["ticker"]
            target = w["target_entry_price"]
            try:
                q = market_data.get_stock_quote(ticker)
                price = q.get("price")
                if price is None:
                    continue
                away_pct = (price - target) / target * 100
                if away_pct <= 0:
                    triggered.append((ticker, price, target, away_pct, w.get("reason", "")[:50]))
                    alert_count += 1
                elif away_pct <= 5:
                    approaching.append((ticker, price, target, away_pct, w.get("reason", "")[:50]))
            except Exception:
                continue

        console.print("[bold]WATCHLIST TRIGGERS[/bold]")
        if triggered:
            for ticker, price, target, away, reason in triggered:
                console.print(
                    f"  [bold green]▶ AT TARGET[/bold green]  [bold]{ticker}[/bold]  "
                    f"${price:.2f} ≤ ${target:.2f}  ({away:+.1f}%)  {reason}"
                )
        if approaching:
            for ticker, price, target, away, reason in approaching:
                console.print(
                    f"  [yellow]⚡ NEAR TARGET[/yellow]  [bold]{ticker}[/bold]  "
                    f"${price:.2f} vs ${target:.2f}  ({away:+.1f}%)  {reason}"
                )
        if not triggered and not approaching:
            console.print("  [dim]No watchlist items at or near target[/dim]")
        console.print()

    # ── 3. Large moves in held positions ──────────────────────────────────────
    if holdings:
        console.print("[bold]HOLDINGS — LARGE MOVES (>10% from cost)[/bold]")
        movers = []
        for h in holdings:
            ticker = h["ticker"]
            avg_cost = h["avg_cost"]
            try:
                q = market_data.get_stock_quote(ticker)
                price = q.get("price")
                if price is None:
                    continue
                move = (price - avg_cost) / avg_cost * 100
                if abs(move) >= 10:
                    movers.append((ticker, price, avg_cost, move))
            except Exception:
                continue

        if movers:
            for ticker, price, cost, move in sorted(movers, key=lambda x: abs(x[3]), reverse=True):
                col = "green" if move > 0 else "red"
                icon = "↑" if move > 0 else "↓"
                console.print(
                    f"  [{col}]{icon}[/{col}] [bold]{ticker}[/bold]  "
                    f"${price:.2f}  cost ${cost:.2f}  [{col}]{move:+.1f}%[/{col}]"
                )
                if abs(move) >= 20:
                    alert_count += 1
        else:
            console.print("  [dim]No positions with moves ≥10% from cost[/dim]")
        console.print()

    # ── Summary ───────────────────────────────────────────────────────────────
    if alert_count > 0:
        console.print(Panel(
            f"[bold yellow]{alert_count} actionable alert(s) found.[/bold yellow] "
            "Run a full agent session to investigate.",
            title="ACTION REQUIRED", border_style="yellow"
        ))
    else:
        console.print("[dim]No actionable alerts. All clear.[/dim]")

    return alert_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Investment watchlist monitor")
    parser.add_argument("--watch", action="store_true",
                        help="Refresh on a loop instead of one-shot")
    parser.add_argument("--interval", type=int, default=60,
                        help="Refresh interval in seconds (default: 60)")
    parser.add_argument("--alert", action="store_true",
                        help="Lightweight alert check: 8-K filings + price triggers (no full render)")
    parser.add_argument("--since", type=str, default=None,
                        help="Check 8-K filings since this date (YYYY-MM-DD). Default: last 7 days.")
    args = parser.parse_args()

    if args.alert:
        n = check_alerts(since=args.since)
        sys.exit(1 if n > 0 else 0)  # non-zero exit when alerts exist (useful for cron jobs)
    elif args.watch:
        try:
            while True:
                console.clear()
                render(console)
                console.print(f"\n[dim]Next refresh in {args.interval}s — Ctrl-C to stop[/dim]")
                time.sleep(args.interval)
        except KeyboardInterrupt:
            console.print("\n[dim]Monitor stopped.[/dim]")
    else:
        render(console)


if __name__ == "__main__":
    main()
