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


def print_reflections(reflections: list[dict]) -> None:
    """Display saved post-session reflections."""
    if not reflections:
        console.print("[dim]No reflections saved yet. They are written by the agent at the end of each review.[/dim]")
        return

    for i, r in enumerate(reflections, 1):
        date = r.get("created_at", "")[:16].replace("T", " ")
        value = r.get("portfolio_value")
        value_str = f"  Portfolio: [yellow]${value:,.0f}[/yellow]" if value else ""
        console.print(
            Panel(
                r.get("reflection", ""),
                title=f"[bold cyan]Reflection {i}[/bold cyan]  [dim]{date} UTC[/dim]{value_str}",
                box=box.ROUNDED,
            )
        )


def print_theses(memory: dict) -> None:
    """Display investment theses for current holdings and recently closed positions."""
    holdings = memory.get("current_holdings_theses", [])
    closed = memory.get("recently_closed_positions", [])

    if holdings:
        console.print("\n[bold magenta]Current Holdings — Buy Theses[/bold magenta]")
        table = Table(box=box.SIMPLE_HEAVY, header_style="bold magenta", show_lines=True)
        table.add_column("Ticker", style="bold cyan", width=8)
        table.add_column("Avg Cost", justify="right", width=10)
        table.add_column("Bought", width=12)
        table.add_column("Original Buy Thesis", width=70)

        for h in holdings:
            date = (h.get("thesis_date") or h.get("first_bought") or "")[:10]
            table.add_row(
                h["ticker"],
                f"${h['avg_cost_per_share']:.2f}",
                date,
                h.get("original_buy_thesis") or "[dim](none recorded)[/dim]",
            )
        console.print(table)
    else:
        console.print("[dim]No current holdings.[/dim]")

    if closed:
        console.print("\n[bold magenta]Recently Closed Positions[/bold magenta]")
        table2 = Table(box=box.SIMPLE_HEAVY, header_style="bold magenta", show_lines=True)
        table2.add_column("Ticker", style="bold cyan", width=8)
        table2.add_column("Sold", width=12)
        table2.add_column("Realized P&L", justify="right", width=14)
        table2.add_column("Sell Rationale", width=60)

        for c in closed:
            pnl = c.get("realized_pnl")
            if pnl is not None:
                color = "green" if pnl >= 0 else "red"
                sign = "+" if pnl >= 0 else ""
                pnl_str = f"[{color}]{sign}${pnl:,.2f}[/{color}]"
            else:
                pnl_str = "[dim]N/A[/dim]"
            sold_date = (c.get("sold_at") or "")[:10]
            rationale = c.get("sell_thesis") or c.get("sell_notes") or "[dim](none recorded)[/dim]"
            table2.add_row(c["ticker"], sold_date, pnl_str, str(rationale)[:60])
        console.print(table2)


def print_benchmark(data: dict) -> None:
    """Display portfolio vs S&P 500 benchmark comparison."""
    if "note" in data:
        console.print(f"[dim]{data['note']}[/dim]")
        return

    port_ret = data.get("portfolio_return_pct", 0)
    bench_ret = data.get("benchmark_return_pct")
    alpha = data.get("alpha_pct")
    is_beating = data.get("is_beating_benchmark")

    port_color = "green" if port_ret >= 0 else "red"
    port_sign = "+" if port_ret >= 0 else ""

    lines = [
        f"[bold]Since:[/bold]          {data.get('start_date', 'N/A')}",
        f"[bold]Portfolio:[/bold]      [yellow]${data.get('portfolio_start_value', 0):>12,.0f}[/yellow]  →  "
        f"[yellow]${data.get('portfolio_current_value', 0):>12,.0f}[/yellow]  "
        f"[{port_color}]({port_sign}{port_ret:.2f}%)[/{port_color}]",
    ]

    if bench_ret is not None:
        bench_color = "green" if bench_ret >= 0 else "red"
        bench_sign = "+" if bench_ret >= 0 else ""
        alpha_color = "green" if (alpha or 0) >= 0 else "red"
        alpha_sign = "+" if (alpha or 0) >= 0 else ""
        beating_str = "[green]YES — beating the index[/green]" if is_beating else "[red]NO — trailing the index[/red]"

        lines += [
            f"[bold]S&P 500:[/bold]        ${data.get('benchmark_start_price', 0):>12,.0f}  →  "
            f"${data.get('benchmark_current_price', 0):>12,.0f}  "
            f"[{bench_color}]({bench_sign}{bench_ret:.2f}%)[/{bench_color}]",
            f"[bold]Alpha:[/bold]          [{alpha_color}]{alpha_sign}{alpha:.2f}%[/{alpha_color}]",
            f"[bold]Beating index:[/bold]  {beating_str}",
        ]
    else:
        lines.append("[dim]Benchmark return unavailable (no S&P 500 price at first snapshot)[/dim]")

    console.print(Panel("\n".join(lines), title="[bold]Portfolio vs S&P 500[/bold]", box=box.ROUNDED))

    recent = data.get("recent_history", [])
    if recent:
        table = Table(box=box.SIMPLE, header_style="bold blue", title="Recent Snapshots")
        table.add_column("Date", width=12)
        table.add_column("Portfolio Value", justify="right", width=16)
        table.add_column("S&P 500", justify="right", width=12)
        for snap in recent:
            spy = snap.get("sp500_price")
            table.add_row(
                snap["date"],
                f"${snap['portfolio_value']:,.0f}",
                f"${spy:,.0f}" if spy else "N/A",
            )
        console.print(table)
