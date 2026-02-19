#!/usr/bin/env python3
"""
Weekly portfolio review scheduler.

Runs the AI investment agent on a recurring schedule (default: every Monday at 09:00).

Usage:
    python scheduler.py            # Start the weekly scheduler (runs forever)
    python scheduler.py --now      # Run one review immediately, then start the schedule
    python scheduler.py --once     # Run one review immediately and exit

Configuration via .env:
    SCHEDULE_DAY   = monday        (mon/tue/wed/thu/fri/sat/sun, default: monday)
    SCHEDULE_TIME  = 09:00         (HH:MM 24h, default: 09:00)
"""

import os
import sys
import time
import logging
from datetime import datetime
from pathlib import Path

import schedule
from dotenv import load_dotenv
from rich.console import Console

load_dotenv()

console = Console()

# ── Logging setup ──────────────────────────────────────────────────────────────
LOG_DIR = Path(__file__).parent / "data"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "scheduler.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("scheduler")

# ── Day mapping ────────────────────────────────────────────────────────────────
DAY_MAP = {
    "monday": schedule.every().monday,
    "mon": schedule.every().monday,
    "tuesday": schedule.every().tuesday,
    "tue": schedule.every().tuesday,
    "wednesday": schedule.every().wednesday,
    "wed": schedule.every().wednesday,
    "thursday": schedule.every().thursday,
    "thu": schedule.every().thursday,
    "friday": schedule.every().friday,
    "fri": schedule.every().friday,
    "saturday": schedule.every().saturday,
    "sat": schedule.every().saturday,
    "sunday": schedule.every().sunday,
    "sun": schedule.every().sunday,
}


def run_weekly_review() -> None:
    """Execute one full portfolio review session and log the outcome."""
    started_at = datetime.now()
    log.info("=== Weekly portfolio review starting ===")

    try:
        # Lazy import so scheduler can start without blocking on imports
        from agent.investment_agent import run_portfolio_review
        from agent.portfolio import initialize_portfolio

        starting_cash = float(os.environ.get("STARTING_CASH", "100000"))
        initialize_portfolio(starting_cash)

        def on_tool_call(name: str, inp: dict) -> None:
            ticker = inp.get("ticker", inp.get("query", ""))
            log.info(f"  tool={name} {ticker}")

        response = run_portfolio_review(on_tool_call=on_tool_call)

        elapsed = (datetime.now() - started_at).total_seconds()
        log.info(f"=== Review complete in {elapsed:.1f}s ===")
        log.info(f"Agent summary: {response[:300]}")

    except Exception as exc:
        log.exception(f"Review failed: {exc}")


def _check_api_key() -> bool:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        console.print(
            "[red]Error:[/red] ANTHROPIC_API_KEY not set in .env"
        )
        return False
    return True


def _schedule_weekly() -> None:
    day_str = os.environ.get("SCHEDULE_DAY", "monday").lower().strip()
    time_str = os.environ.get("SCHEDULE_TIME", "09:00").strip()

    day_schedule = DAY_MAP.get(day_str)
    if day_schedule is None:
        console.print(
            f"[red]Error:[/red] Unknown SCHEDULE_DAY '{day_str}'. "
            f"Use: monday, tuesday, wednesday, thursday, friday, saturday, sunday"
        )
        sys.exit(1)

    day_schedule.at(time_str).do(run_weekly_review)

    next_run = schedule.next_run()
    console.print(
        f"\n[bold green]Scheduler started.[/bold green]\n"
        f"  Runs every [bold]{day_str.capitalize()}[/bold] at [bold]{time_str}[/bold]\n"
        f"  Next run: [yellow]{next_run}[/yellow]\n"
        f"  Log file: [dim]{LOG_FILE}[/dim]\n"
        f"  Press Ctrl+C to stop.\n"
    )
    log.info(f"Scheduler started — runs every {day_str} at {time_str}, next: {next_run}")


def main() -> None:
    if not _check_api_key():
        sys.exit(1)

    args = sys.argv[1:]

    if "--once" in args:
        # Run once immediately and exit
        console.print("[bold]Running one-off portfolio review...[/bold]")
        run_weekly_review()
        return

    _schedule_weekly()

    if "--now" in args:
        # Run immediately, then continue on schedule
        console.print("[dim]Running immediate review before starting schedule...[/dim]")
        run_weekly_review()

    # Main scheduler loop
    try:
        while True:
            schedule.run_pending()
            time.sleep(30)  # Check every 30 seconds
    except KeyboardInterrupt:
        console.print("\n[dim]Scheduler stopped.[/dim]")
        log.info("Scheduler stopped by user.")


if __name__ == "__main__":
    main()
