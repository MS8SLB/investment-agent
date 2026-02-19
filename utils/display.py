"""
Rich-based display helpers for the investment agent CLI.
"""

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box
from typing import Any

console = Console()


def print_header(title: str) -> None:
    console.print(Panel(f"[bold cyan]{title}[/bold cyan]", box=box.DOUBLE_EDGE))


def print_portfolio(status: dict) -> None:
    """Pretty-print portfolio status."""
    cash = status.get("cash", 0)
    total = status.get("total_portfolio_value", 0)
    invested = status.get("total_invested_value", 0)
    unrealized = status.get("total_unrealized_pnl", 0)
    holdings = status.get("holdings", [])

    # Summary panel
    pnl_color = "green" if unrealized >= 0 else "red"
    pnl_sign = "+" if unrealized >= 0 else ""
    summary = (
        f"[bold]Total Value:[/bold]  [yellow]${total:>12,.2f}[/yellow]\n"
        f"[bold]Cash:[/bold]         [white]${cash:>12,.2f}[/white]\n"
        f"[bold]Invested:[/bold]     [white]${invested:>12,.2f}[/white]\n"
        f"[bold]Unrealized P&L:[/bold] [{pnl_color}]{pnl_sign}${unrealized:>10,.2f}[/{pnl_color}]"
    )
    console.print(Panel(summary, title="[bold]Portfolio Summary[/bold]", box=box.ROUNDED))

    if not holdings:
        console.print("[dim]No holdings yet.[/dim]")
        return

    # Holdings table
    table = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold magenta")
    table.add_column("Ticker", style="bold cyan", width=8)
    table.add_column("Shares", justify="right", width=10)
    table.add_column("Avg Cost", justify="right", width=10)
    table.add_column("Price", justify="right", width=10)
    table.add_column("Mkt Value", justify="right", width=12)
    table.add_column("P&L", justify="right", width=12)
    table.add_column("P&L %", justify="right", width=8)

    for h in holdings:
        pnl = h.get("unrealized_pnl")
        pct = h.get("unrealized_pct")
        if pnl is not None:
            color = "green" if pnl >= 0 else "red"
            sign = "+" if pnl >= 0 else ""
            pnl_str = f"[{color}]{sign}${pnl:,.2f}[/{color}]"
            pct_str = f"[{color}]{sign}{pct:.2f}%[/{color}]"
        else:
            pnl_str = "[dim]N/A[/dim]"
            pct_str = "[dim]N/A[/dim]"

        price = h.get("current_price")
        mv = h.get("market_value")

        table.add_row(
            h["ticker"],
            f"{h['shares']:.4f}",
            f"${h['avg_cost']:.2f}",
            f"${price:.2f}" if price else "N/A",
            f"${mv:,.2f}" if mv else "N/A",
            pnl_str,
            pct_str,
        )

    console.print(table)


def print_transactions(transactions: list[dict]) -> None:
    if not transactions:
        console.print("[dim]No transactions yet.[/dim]")
        return

    table = Table(box=box.SIMPLE, header_style="bold blue", title="Recent Transactions")
    table.add_column("Date", width=19)
    table.add_column("Action", width=6)
    table.add_column("Ticker", width=8)
    table.add_column("Shares", justify="right", width=10)
    table.add_column("Price", justify="right", width=10)
    table.add_column("Total", justify="right", width=12)
    table.add_column("Notes", width=40)

    for tx in transactions:
        action_color = "green" if tx["action"] == "BUY" else "red"
        table.add_row(
            tx["ts"][:19],
            f"[{action_color}]{tx['action']}[/{action_color}]",
            tx["ticker"],
            f"{tx['shares']:.4f}",
            f"${tx['price']:.2f}",
            f"${tx['total']:,.2f}",
            (tx.get("notes") or "")[:40],
        )

    console.print(table)


def print_market_summary(summary: dict) -> None:
    table = Table(box=box.SIMPLE, header_style="bold blue", title="Market Summary")
    table.add_column("Index", width=16)
    table.add_column("Price", justify="right", width=12)
    table.add_column("Change", justify="right", width=10)

    for name, data in summary.items():
        price = data.get("price")
        change = data.get("change_pct")
        if change is not None:
            color = "green" if change >= 0 else "red"
            sign = "+" if change >= 0 else ""
            change_str = f"[{color}]{sign}{change:.2f}%[/{color}]"
        else:
            change_str = "[dim]N/A[/dim]"

        table.add_row(
            name,
            f"${price:,.2f}" if price else "N/A",
            change_str,
        )

    console.print(table)


def print_agent_thinking(tool_name: str, tool_input: dict) -> None:
    console.print(f"  [dim cyan]⚙ Tool:[/dim cyan] [bold]{tool_name}[/bold]", highlight=False)
    key_args = {k: v for k, v in tool_input.items() if k != "notes" and v is not None}
    if key_args:
        args_str = ", ".join(f"{k}={v}" for k, v in key_args.items())
        console.print(f"    [dim]{args_str}[/dim]")


def print_tool_result_summary(tool_name: str, result: Any) -> None:
    if isinstance(result, dict):
        if "error" in result:
            console.print(f"    [red]Error: {result['error']}[/red]")
        elif tool_name == "buy_stock" and result.get("success"):
            console.print(
                f"    [green]Bought {result['shares']:.4f} shares of {result['ticker']} "
                f"@ ${result['price']:.2f} (Total: ${result['total_cost']:,.2f})[/green]"
            )
        elif tool_name == "sell_stock" and result.get("success"):
            pnl = result.get("realized_pnl", 0)
            color = "green" if pnl >= 0 else "red"
            sign = "+" if pnl >= 0 else ""
            console.print(
                f"    [{color}]Sold {result['shares']:.4f} shares of {result['ticker']} "
                f"@ ${result['price']:.2f} | P&L: {sign}${pnl:.2f}[/{color}]"
            )
        elif tool_name == "get_stock_quote" and "price" in result:
            console.print(
                f"    [dim]{result.get('name', '')} ({result['ticker']}): "
                f"${result['price']:.2f}[/dim]"
            )
