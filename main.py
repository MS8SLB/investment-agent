#!/usr/bin/env python3
"""
AI Investment Portfolio Agent - CLI Entry Point

Usage:
    python main.py                    # Interactive menu
    python main.py review             # Run a full AI portfolio review
    python main.py portfolio          # Show portfolio status
    python main.py history            # Show transaction history
    python main.py market             # Show market summary
    python main.py benchmark          # Portfolio vs S&P 500 performance
    python main.py reflections        # Show agent's past session reflections
    python main.py theses             # Show investment theses for holdings
    python main.py ask "your prompt"  # Ask the agent anything
"""

import os
import sys

from dotenv import load_dotenv
from rich.console import Console
from rich.prompt import Prompt

load_dotenv()

console = Console()


def _check_api_key() -> bool:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        console.print(
            "[red]Error:[/red] ANTHROPIC_API_KEY not set.\n"
            "Copy [bold].env.example[/bold] to [bold].env[/bold] and add your API key."
        )
        return False
    return True


def _init_portfolio() -> None:
    starting_cash = float(os.environ.get("STARTING_CASH", "100000"))
    from agent.portfolio import initialize_portfolio
    initialize_portfolio(starting_cash)


def cmd_portfolio() -> None:
    """Display current portfolio status."""
    from agent.tools import _get_portfolio_status
    from utils.display import print_header, print_portfolio

    print_header("Current Portfolio")
    console.print("[dim]Fetching live prices...[/dim]")
    status = _get_portfolio_status()
    print_portfolio(status)


def cmd_history() -> None:
    """Display recent transaction history."""
    from agent.portfolio import get_transactions
    from utils.display import print_header, print_transactions

    print_header("Transaction History")
    txs = get_transactions(30)
    print_transactions(txs)


def cmd_market() -> None:
    """Display market summary."""
    from agent.market_data import get_market_summary
    from utils.display import print_header, print_market_summary

    print_header("Market Summary")
    console.print("[dim]Fetching market data...[/dim]")
    summary = get_market_summary()
    print_market_summary(summary)


def cmd_benchmark() -> None:
    """Display portfolio performance vs S&P 500."""
    from agent.tools import _handle_benchmark_comparison
    from utils.display import print_header, print_benchmark

    print_header("Portfolio vs Benchmark")
    data = _handle_benchmark_comparison()
    print_benchmark(data)


def cmd_reflections() -> None:
    """Display the agent's saved post-session reflections."""
    from agent.portfolio import get_reflections
    from utils.display import print_header, print_reflections

    print_header("Agent Reflections")
    reflections = get_reflections(limit=10)
    print_reflections(reflections)


def cmd_theses() -> None:
    """Display investment theses for current holdings and closed positions."""
    from agent.portfolio import get_investment_memory
    from utils.display import print_header, print_theses

    print_header("Investment Theses")
    memory = get_investment_memory()
    print_theses(memory)


def cmd_review() -> None:
    """Run a full autonomous portfolio review."""
    if not _check_api_key():
        return

    from agent.investment_agent import run_portfolio_review
    from utils.display import print_header, print_agent_thinking, print_tool_result_summary

    print_header("AI Portfolio Review")
    console.print("[dim]The agent is analyzing markets and managing your portfolio...[/dim]\n")

    def on_text(text: str) -> None:
        console.print(text, markup=False)

    def on_tool_call(name: str, inp: dict) -> None:
        print_agent_thinking(name, inp)

    def on_tool_result(name: str, result) -> None:
        print_tool_result_summary(name, result)

    response = run_portfolio_review(
        on_text=on_text,
        on_tool_call=on_tool_call,
        on_tool_result=on_tool_result,
    )

    console.print("\n[bold green]Review complete.[/bold green]")


def cmd_ask(prompt: str) -> None:
    """Ask the agent a custom question or give a custom instruction."""
    if not _check_api_key():
        return

    from agent.investment_agent import run_custom_prompt
    from utils.display import print_agent_thinking, print_tool_result_summary

    console.print(f"\n[dim]Asking agent: {prompt}[/dim]\n")

    def on_text(text: str) -> None:
        console.print(text, markup=False)

    def on_tool_call(name: str, inp: dict) -> None:
        print_agent_thinking(name, inp)

    def on_tool_result(name: str, result) -> None:
        print_tool_result_summary(name, result)

    run_custom_prompt(
        prompt,
        on_text=on_text,
        on_tool_call=on_tool_call,
        on_tool_result=on_tool_result,
    )


def cmd_interactive() -> None:
    """Interactive menu loop."""
    from utils.display import print_header

    print_header("AI Investment Portfolio Agent")
    console.print(
        "\n[bold]Paper trading portfolio powered by Claude AI.[/bold]\n"
        "Your portfolio starts with [yellow]$100,000[/yellow] in virtual cash.\n"
    )

    while True:
        console.print("\n[bold cyan]What would you like to do?[/bold cyan]")
        console.print("  [1] View portfolio")
        console.print("  [2] Run AI portfolio review")
        console.print("  [3] Ask the agent something")
        console.print("  [4] View transaction history")
        console.print("  [5] View market summary")
        console.print("  [6] Portfolio vs benchmark (S&P 500)")
        console.print("  [7] View agent reflections")
        console.print("  [8] View investment theses")
        console.print("  [q] Quit")

        choice = Prompt.ask("\nChoice", choices=["1", "2", "3", "4", "5", "6", "7", "8", "q"], default="1")

        if choice == "1":
            cmd_portfolio()
        elif choice == "2":
            cmd_review()
        elif choice == "3":
            prompt = Prompt.ask("What would you like to ask the agent?")
            if prompt.strip():
                cmd_ask(prompt.strip())
        elif choice == "4":
            cmd_history()
        elif choice == "5":
            cmd_market()
        elif choice == "6":
            cmd_benchmark()
        elif choice == "7":
            cmd_reflections()
        elif choice == "8":
            cmd_theses()
        elif choice == "q":
            console.print("[dim]Goodbye![/dim]")
            break


def main() -> None:
    _init_portfolio()

    args = sys.argv[1:]

    if not args:
        cmd_interactive()
    elif args[0] == "review":
        cmd_review()
    elif args[0] == "portfolio":
        cmd_portfolio()
    elif args[0] == "history":
        cmd_history()
    elif args[0] == "market":
        cmd_market()
    elif args[0] == "benchmark":
        cmd_benchmark()
    elif args[0] == "reflections":
        cmd_reflections()
    elif args[0] == "theses":
        cmd_theses()
    elif args[0] == "ask" and len(args) > 1:
        cmd_ask(" ".join(args[1:]))
    else:
        console.print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
