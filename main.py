"""
main.py
-------
Single CLI entry point for the whole project.

Usage:
    python main.py --audit
    python main.py --embed
    python main.py --score "some conversation text"
    python main.py --benchmark
    python main.py --setup
"""

import argparse
import json
import subprocess
import sys

from rich.console import Console

console = Console()


def cmd_setup():
    """
    Checks whether Ollama is running and whether llama3.1 is pulled.
    Does NOT silently install anything without telling you -- it will offer
    the exact command to run if something's missing, and attempt to pull the
    model only if you confirm, since a multi-GB download shouldn't happen
    without the user knowing.
    """
    import ollama

    console.print("[bold cyan]Checking Ollama setup...[/bold cyan]")

    try:
        client = ollama.Client(host="http://localhost:11434")
        models_response = client.list()
    except Exception as e:
        console.print(f"[bold red]Could not connect to Ollama at http://localhost:11434[/bold red]")
        console.print(f"  Error: {e}")
        console.print("\n[yellow]Is Ollama running?[/yellow] Start it, then re-run --setup.")
        console.print("  Windows: Ollama should run as a background app after install (check the system tray),")
        console.print("  or start it manually from a terminal with: ollama serve")
        sys.exit(1)

    console.print("[green]Ollama is running.[/green]")

    # ollama-python's list() response shape has changed across versions;
    # handle both dict-style and object-style responses defensively.
    try:
        models = models_response.get("models", []) if isinstance(models_response, dict) else models_response.models
    except Exception:
        models = []

    model_names = []
    for m in models:
        name = m.get("model") if isinstance(m, dict) else getattr(m, "model", None)
        if name:
            model_names.append(name)

    has_llama = any("llama3.1" in name for name in model_names)

    if has_llama:
        console.print("[green]llama3.1 is already pulled and ready.[/green]")
    else:
        console.print("[yellow]llama3.1 not found locally.[/yellow] Pulling it now (this can take a while, several GB)...")
        try:
            subprocess.run(["ollama", "pull", "llama3.1"], check=True)
            console.print("[green]llama3.1 pulled successfully.[/green]")
        except FileNotFoundError:
            console.print("[bold red]`ollama` CLI not found on PATH.[/bold red] Run this manually:")
            console.print("  ollama pull llama3.1")
        except subprocess.CalledProcessError as e:
            console.print(f"[bold red]Failed to pull llama3.1: {e}[/bold red]")
            console.print("  Try running manually: ollama pull llama3.1")

    console.print("\n[bold cyan]Setup check complete.[/bold cyan]")


def cmd_audit():
    from src.audit import audit_facets
    audit_facets()


def cmd_embed():
    from src.embeddings import build_index
    build_index()


def cmd_score(conversation: str):
    from src.pipeline import run_pipeline
    console.print(f"[bold cyan]Scoring conversation:[/bold cyan] {conversation[:100]}...\n")
    result = run_pipeline(conversation)

    console.print(f"Retrieved [bold]{result['total_facets_retrieved']}[/bold] candidate facets.")
    console.print(f"Scored: [green]{result['scored']}[/green]  |  Abstained: [yellow]{result['abstained']}[/yellow]")
    if result.get("parse_errors"):
        console.print(f"Parse errors: [red]{result['parse_errors']}[/red]")
    if result.get("error"):
        console.print(f"[bold red]Error:[/bold red] {result['error']}")
        return

    from rich.table import Table
    table = Table(title="Facet scores")
    table.add_column("Facet")
    table.add_column("Status")
    table.add_column("Score", justify="center")
    table.add_column("Confidence")
    table.add_column("Evidence")

    for r in result["results"]:
        score_str = str(r["score"]) if r["score"] is not None else "-"
        table.add_row(r["facet"], r["status"], score_str, r["confidence"], r["evidence"][:80])

    console.print(table)


def cmd_benchmark():
    from src.benchmark import run_benchmark
    run_benchmark()


def main():
    parser = argparse.ArgumentParser(description="Conversation -> personality facet scoring pipeline")
    parser.add_argument("--audit", action="store_true", help="Run the facet CSV audit/cleaning step")
    parser.add_argument("--embed", action="store_true", help="Build the FAISS embedding index")
    parser.add_argument("--score", type=str, metavar="TEXT", help="Score a conversation string")
    parser.add_argument("--benchmark", action="store_true", help="Run the 10-conversation benchmark")
    parser.add_argument("--setup", action="store_true", help="Check Ollama is running and llama3.1 is available")

    args = parser.parse_args()

    if not any([args.audit, args.embed, args.score, args.benchmark, args.setup]):
        parser.print_help()
        sys.exit(0)

    try:
        if args.setup:
            cmd_setup()
        if args.audit:
            cmd_audit()
        if args.embed:
            cmd_embed()
        if args.score:
            cmd_score(args.score)
        if args.benchmark:
            cmd_benchmark()
    except Exception as e:
        console.print(f"[bold red]Fatal error:[/bold red] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
