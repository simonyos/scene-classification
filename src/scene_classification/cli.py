"""Typer CLI entry points."""

from __future__ import annotations

import typer
from rich import print as rprint

from scene_classification.config import Settings

app = typer.Typer(add_completion=False, no_args_is_help=True, help="Scene classification CLI.")


@app.command("prepare-data")
def prepare_data(
    classes: str = typer.Option(
        "bedroom,bathroom,livingroom,kitchen",
        help="Comma-separated MIT Indoor67 class names.",
    ),
    train: float = 0.7,
    val: float = 0.15,
    test: float = 0.15,
) -> None:
    """Download MIT Indoor67 and build stratified train/val/test splits."""
    from scene_classification.data.download import prepare_dataset

    settings = Settings()
    counts = prepare_dataset(settings, classes.split(","), (train, val, test))
    rprint({"splits": counts})


@app.command("extract-features")
def extract_features() -> None:
    """Run YOLOv8 over all splits and save features.csv."""
    from scene_classification.features.extract import extract_all

    out = extract_all(Settings())
    rprint(f"Wrote [bold]{out}[/bold]")


@app.command("train-tabular")
def train_tabular() -> None:
    """Train DT/KNN/NB/SVM/LR/XGB on YOLO features; log to MLflow."""
    from scene_classification.models.train_tabular import run

    results = run(Settings())
    for r in results:
        rprint(
            f"{r.name}: cv={r.cv_accuracy:.3f} val={r.val_accuracy:.3f} "
            f"test={r.test_accuracy:.3f} f1={r.test_macro_f1:.3f}"
        )


@app.command("train-cnn")
def train_cnn() -> None:
    """Train a transfer-learned ResNet50 baseline."""
    from scene_classification.models.train_cnn import run

    result = run(Settings())
    rprint({"val_accuracy": result.val_accuracy, "test_accuracy": result.test_accuracy})


@app.command("evaluate")
def evaluate() -> None:
    """Build confusion matrices, comparison chart, summary.md."""
    from scene_classification.evaluate import run

    out = run(Settings())
    rprint(f"Wrote [bold]{out}[/bold]")


if __name__ == "__main__":
    app()
