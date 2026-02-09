from __future__ import annotations

import json
import typer
from rich import print
from rich.table import Table

from .train import train as train_fn
from .recommend import recommend_by_title
from .evaluate import retrieval_self_check, save_report

app = typer.Typer(add_completion=False, help="Movie Recommender (Content-Based) — Python + Keras")

@app.command()
def train(
    csv: str = typer.Option(..., "--csv", help="Path to movies CSV (e.g., data/movies.csv)"),
    out_dir: str = typer.Option("models", "--out-dir", help="Where to save model + embeddings"),
    n_neg: int = typer.Option(3, "--n-neg", help="Negatives per positive pair"),
    batch_size: int = typer.Option(256, "--batch-size"),
    epochs: int = typer.Option(15, "--epochs"),
    seed: int = typer.Option(42, "--seed"),
):
    """Train the siamese model and export embeddings."""
    report = train_fn(
        csv_path=csv,
        out_dir=out_dir,
        n_neg=n_neg,
        batch_size=batch_size,
        epochs=epochs,
        seed=seed,
    )
    print("[bold green]Training complete.[/bold green]")
    print(json.dumps(report, indent=2))

@app.command()
def recommend(
    title: str = typer.Option(..., "--title", help="Movie title (exact or partial match)"),
    k: int = typer.Option(10, "--k", help="Number of recommendations"),
    models_dir: str = typer.Option("models", "--models-dir"),
):
    """Recommend movies similar to the given title."""
    res = recommend_by_title(title=title, k=k, models_dir=models_dir)
    if res is None:
        print("[bold red]Movie not found.[/bold red]")
        raise typer.Exit(code=1)

    table = Table(title=f"Top {k} recommendations")
    table.add_column("Query")
    table.add_column("Recommended Title")
    table.add_column("Genres")
    table.add_column("Score", justify="right")
    for _, row in res.iterrows():
        table.add_row(str(row["query_title"]), str(row["title"]), str(row["genres"])[:60], f"{row['score']:.4f}")
    print(table)

@app.command()
def evaluate(
    k: int = typer.Option(10, "--k", help="Top-K threshold for self-retrieval"),
    models_dir: str = typer.Option("models", "--models-dir"),
    out_csv: str = typer.Option("outputs/eval_report.csv", "--out-csv"),
):
    """Run a lightweight retrieval sanity-check evaluation."""
    report = retrieval_self_check(models_dir=models_dir, k=k)
    save_report(report, out_csv)
    print("[bold cyan]Evaluation report:[/bold cyan]")
    print(json.dumps(report, indent=2))
    print(f"[dim]Saved to {out_csv}[/dim]")

if __name__ == "__main__":
    app()
