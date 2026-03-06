# src/disinformation_classification/__main__.py

"""
Entry point for running the disinformation_classification package as a module.

Allows you to run:
    python -m disinformation_classification train
"""

from .cli import app  # import the Typer app

# Run the Typer app
app()