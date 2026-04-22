"""
Transfer-learned CNN baseline (ResNet50, ImageNet weights).

Trains the classifier head (and optionally fine-tunes late layers), early-stops on val
accuracy. Experiments tracked to MLflow.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import mlflow
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

from scene_classification.config import Settings


@dataclass
class CNNConfig:
    arch: str = "resnet50"
    batch_size: int = 32
    epochs: int = 15
    lr_head: float = 1e-3
    lr_backbone: float = 1e-4
    weight_decay: float = 1e-4
    unfreeze_last_block: bool = True
    patience: int = 4


@dataclass
class CNNResult:
    val_accuracy: float
    test_accuracy: float
    train_seconds: float
    weights_path: Path


def _pick_device(pref: str) -> torch.device:
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _build_dataloaders(splits_root: Path, batch_size: int):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    train_ds = datasets.ImageFolder(splits_root / "train", transform=train_tf)
    val_ds = datasets.ImageFolder(splits_root / "val", transform=eval_tf)
    test_ds = datasets.ImageFolder(splits_root / "test", transform=eval_tf)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True),
        train_ds.classes,
    )


def _build_model(arch: str, num_classes: int, unfreeze_last_block: bool) -> nn.Module:
    if arch != "resnet50":
        raise ValueError(f"Unsupported arch: {arch}")
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    for p in model.parameters():
        p.requires_grad = False
    if unfreeze_last_block:
        for p in model.layer4.parameters():
            p.requires_grad = True
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


@torch.no_grad()
def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / max(total, 1)


def run(settings: Settings, cfg: CNNConfig | None = None) -> CNNResult:
    cfg = cfg or CNNConfig()
    splits_root = settings.processed_dir / "splits"
    device = _pick_device(settings.device)

    train_loader, val_loader, test_loader, classes = _build_dataloaders(splits_root, cfg.batch_size)
    model = _build_model(cfg.arch, len(classes), cfg.unfreeze_last_block).to(device)

    params = [
        {"params": [p for p in model.fc.parameters() if p.requires_grad], "lr": cfg.lr_head},
        {
            "params": [p for p in model.layer4.parameters() if p.requires_grad],
            "lr": cfg.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(params, weight_decay=cfg.weight_decay)
    criterion = nn.CrossEntropyLoss()

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment("scene-classification/cnn")
    weights_path = settings.artifacts_root / "cnn_best.pt"
    settings.artifacts_root.mkdir(parents=True, exist_ok=True)

    best_val, best_epoch, patience_left = 0.0, -1, cfg.patience
    t0 = time.perf_counter()

    with mlflow.start_run(run_name=cfg.arch):
        mlflow.log_params(cfg.__dict__ | {"num_classes": len(classes)})
        for epoch in range(cfg.epochs):
            model.train()
            running = 0.0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                running += loss.item() * y.size(0)

            train_loss = running / len(train_loader.dataset)
            val_acc = _evaluate(model, val_loader, device)
            mlflow.log_metrics({"train_loss": train_loss, "val_accuracy": val_acc}, step=epoch)

            if val_acc > best_val:
                best_val, best_epoch, patience_left = val_acc, epoch, cfg.patience
                torch.save(
                    {"state_dict": model.state_dict(), "classes": classes, "arch": cfg.arch},
                    weights_path,
                )
            else:
                patience_left -= 1
                if patience_left <= 0:
                    break

        ckpt = torch.load(weights_path, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        test_acc = _evaluate(model, test_loader, device)
        train_seconds = time.perf_counter() - t0
        mlflow.log_metrics({
            "best_val_accuracy": best_val,
            "best_epoch": best_epoch,
            "test_accuracy": test_acc,
            "train_seconds": train_seconds,
        })
        mlflow.log_artifact(str(weights_path))

    return CNNResult(
        val_accuracy=best_val,
        test_accuracy=test_acc,
        train_seconds=train_seconds,
        weights_path=weights_path,
    )
