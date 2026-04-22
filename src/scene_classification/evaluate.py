"""
Evaluation and figure generation.

Produces:
  reports/figures/confusion_<model>.png
  reports/figures/comparison_bar.png
  reports/summary.md
"""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from scene_classification.config import Settings


def _feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith(("count_", "confsum_", "areafrac_"))]


def _load_test(settings: Settings) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(settings.features_csv)
    test = df[df["split"] == "test"]
    return test[_feature_cols(df)], test["label"]


def _confusion_figure(y_true, y_pred, labels: list[str], out: Path, title: str) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False, xticks_rotation=45)
    ax.set_title(title)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _comparison_bar(summary: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(summary))
    ax.bar(x, summary["test_accuracy"])
    ax.set_xticks(x)
    ax.set_xticklabels(summary["name"], rotation=30, ha="right")
    ax.set_ylabel("Test accuracy")
    ax.set_ylim(0, 1)
    for i, v in enumerate(summary["test_accuracy"]):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def run(settings: Settings) -> Path:
    figs_dir = Path("reports/figures")
    summary_path = settings.artifacts_root / "tabular_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError("Run train-tabular first to produce tabular_summary.csv")
    summary = pd.read_csv(summary_path)

    X_test, y_test = _load_test(settings)
    labels = sorted(y_test.unique())

    for _, row in summary.iterrows():
        model = joblib.load(row["model_path"])
        pred = model.predict(X_test)
        _confusion_figure(
            y_test, pred, labels, figs_dir / f"confusion_{row['name']}.png", row["name"]
        )

    _comparison_bar(summary.sort_values("test_accuracy", ascending=False), figs_dir / "comparison_bar.png")

    md = ["# Results summary", "", summary.to_markdown(index=False)]
    out_md = Path("reports/summary.md")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(md))
    return out_md
