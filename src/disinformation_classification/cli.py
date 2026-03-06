# src/disinformation_classification/cli.py

import typer
from rich.console import Console
from disinformation_classification.train import train_model
from disinformation_classification.predict import predict

app = typer.Typer()
console = Console()


@app.command()
def train():
    """
    Train the disinformation classification model.
    """
    console.print("[bold green]Starting training...[/bold green]")
    train_model()
    console.print("[bold blue]Training finished.[/bold blue]")


@app.command()
def predict_text(text: str):
    """
    Predict whether a text contains disinformation.
    """
    console.print(f"[yellow]Prediction requested for:[/yellow] {text}")
    result = predict(text)  # call your existing predict function
    console.print(f"[bold magenta]Prediction result:[/bold magenta] {result}")


# Optional: keep a simple info command
@app.command()
def info():
    """
    Show CLI information.
    """
    console.print("[bold cyan]Disinformation Classification CLI[/bold cyan]")
    console.print("Available commands: train, predict-text")