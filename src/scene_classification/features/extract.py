"""
YOLOv8-based object feature extraction.

For each image we produce a row with, per COCO class c:
  - count_<c>     : number of detections above conf threshold
  - confsum_<c>   : sum of confidences
  - areafrac_<c>  : sum of (bbox_area / image_area)

This is a richer superset of the original paper's 41-dim count vector,
and still cheap & interpretable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

from scene_classification.config import Settings


@dataclass
class ExtractConfig:
    conf_threshold: float = 0.25
    imgsz: int = 640


def _iter_split_images(split_dir: Path):
    for cls_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        for img_path in sorted(p for p in cls_dir.iterdir() if p.is_file()):
            yield cls_dir.name, img_path


def extract_for_split(
    settings: Settings, split: str, cfg: ExtractConfig | None = None
) -> pd.DataFrame:
    """Extract features for one split ('train' | 'val' | 'test') into a DataFrame."""
    from ultralytics import YOLO  # imported lazily to keep CLI fast

    cfg = cfg or ExtractConfig()
    split_dir = settings.processed_dir / "splits" / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Missing split directory: {split_dir}")

    model = YOLO(settings.yolo_weights)
    class_names: list[str] = [model.names[i] for i in sorted(model.names)]

    rows: list[dict] = []
    for label, img_path in tqdm(list(_iter_split_images(split_dir)), desc=f"extract:{split}"):
        with Image.open(img_path) as im:
            w, h = im.size
        img_area = float(w * h) if w and h else 1.0

        result = model.predict(
            source=str(img_path),
            conf=cfg.conf_threshold,
            imgsz=cfg.imgsz,
            verbose=False,
        )[0]

        counts = {f"count_{c}": 0 for c in class_names}
        confs = {f"confsum_{c}": 0.0 for c in class_names}
        areas = {f"areafrac_{c}": 0.0 for c in class_names}

        if result.boxes is not None and len(result.boxes) > 0:
            cls_ids = result.boxes.cls.cpu().numpy().astype(int)
            conf_vals = result.boxes.conf.cpu().numpy()
            xyxy = result.boxes.xyxy.cpu().numpy()
            for cid, cv, box in zip(cls_ids, conf_vals, xyxy, strict=False):
                name = model.names[int(cid)]
                counts[f"count_{name}"] += 1
                confs[f"confsum_{name}"] += float(cv)
                bw = max(0.0, float(box[2] - box[0]))
                bh = max(0.0, float(box[3] - box[1]))
                areas[f"areafrac_{name}"] += (bw * bh) / img_area

        rows.append({
            "image_path": str(img_path),
            "label": label,
            "split": split,
            **counts,
            **confs,
            **areas,
        })

    return pd.DataFrame(rows)


def extract_all(settings: Settings, cfg: ExtractConfig | None = None) -> Path:
    settings.ensure_dirs()
    frames = [extract_for_split(settings, s, cfg) for s in ("train", "val", "test")]
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(settings.features_csv, index=False)
    return settings.features_csv
