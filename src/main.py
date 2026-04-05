import typer

from .data_cleaning.run_cleaning import run_cleaning
from .data_preprocessing.run_preprocessing import run_preprocessing
from .data_modelling.run_modelling import run_training
from .data_inference.run_inference import run_inference

app = typer.Typer()


@app.command()
def clean():
    run_cleaning()


@app.command()
def preprocess():
    run_preprocessing()


@app.command()
def train():
    run_training()


@app.command()
def infer(text: str):
    run_inference(text)


@app.command()
def full_pipeline():
    run_cleaning()
    run_preprocessing()
    run_training()


if __name__ == "__main__":
    app()