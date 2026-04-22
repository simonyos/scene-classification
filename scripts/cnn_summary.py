"""One-off: reload best CNN weights, compute macro-F1 on test, write cnn_summary.json."""

import json

import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

from scene_classification.config import Settings


def main() -> None:
    s = Settings()
    weights_path = s.artifacts_root / "cnn_best.pt"
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
    ds = datasets.ImageFolder(s.processed_dir / "splits" / "test", transform=tf)
    assert ds.classes == classes, (ds.classes, classes)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)

    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for x, y in loader:
            y_true.extend(y.tolist())
            y_pred.extend(m(x.to(device)).argmax(1).cpu().tolist())

    acc = sum(int(a == b) for a, b in zip(y_true, y_pred, strict=False)) / len(y_true)
    f1 = float(f1_score(y_true, y_pred, average="macro"))

    summary = {
        "name": "cnn_resnet50",
        "val_accuracy": 0.892128279883382,
        "test_accuracy": acc,
        "test_macro_f1": f1,
        "train_seconds": None,
        "classes": classes,
        "weights_path": str(weights_path),
    }
    (s.artifacts_root / "cnn_summary.json").write_text(json.dumps(summary, indent=2))
    print(summary)


if __name__ == "__main__":
    main()
