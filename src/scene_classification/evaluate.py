"""
Evaluation and figure generation.

Produces:
  reports/figures/confusion_<model>.png (tabular + CNN)
  reports/figures/comparison_bar.png
  reports/summary.md
"""

from __future__ import annotations

import json
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


def _cnn_confusion(settings: Settings, labels: list[str], figs_dir: Path) -> None:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from torchvision import datasets, models, transforms

    weights_path = settings.artifacts_root / "cnn_best.pt"
    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    classes = ckpt["classes"]

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    m = models.resnet50(weights=None)
    m.fc = nn.Linear(m.fc.in_features, len(classes))
    m.load_state_dict(ckpt["state_dict"])
    m.to(device).eval()

    tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    ds = datasets.ImageFolder(settings.processed_dir / "splits" / "test", transform=tf)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)

    y_true: list[str] = []
    y_pred: list[str] = []
    with torch.no_grad():
        for x, y in loader:
            yh = m(x.to(device)).argmax(1).cpu().tolist()
            y_true.extend([classes[i] for i in y.tolist()])
            y_pred.extend([classes[i] for i in yh])

    out = figs_dir / "confusion_cnn_resnet50.png"
    _confusion_figure(y_true, y_pred, labels, out, "cnn_resnet50")


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

    cnn_summary_path = settings.artifacts_root / "cnn_summary.json"
    if cnn_summary_path.exists():
        cnn = json.loads(cnn_summary_path.read_text())
        cnn_row = {
            "name": cnn["name"],
            "best_params": "{}",
            "cv_accuracy": np.nan,
            "val_accuracy": cnn["val_accuracy"],
            "test_accuracy": cnn["test_accuracy"],
            "test_macro_f1": cnn["test_macro_f1"],
            "train_seconds": cnn.get("train_seconds"),
            "model_path": cnn["weights_path"],
        }
        summary = pd.concat([summary, pd.DataFrame([cnn_row])], ignore_index=True)
        _cnn_confusion(settings, labels, figs_dir)

    ranked = summary.sort_values("test_accuracy", ascending=False)
    _comparison_bar(ranked, figs_dir / "comparison_bar.png")

    md = ["# Results summary", "", ranked.to_markdown(index=False)]
    out_md = Path("reports/summary.md")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(md))
    return out_md
