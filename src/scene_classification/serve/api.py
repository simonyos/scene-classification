"""
FastAPI inference endpoint.

Loads the best tabular model (per artifacts/best_tabular.txt) + YOLOv8 extractor and
returns predicted class plus the objects detected in the image.
"""

from __future__ import annotations

import io
from functools import lru_cache
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel

from scene_classification.config import Settings


class Prediction(BaseModel):
    label: str
    probabilities: dict[str, float]
    detections: list[dict[str, Any]]


@lru_cache(maxsize=1)
def _settings() -> Settings:
    return Settings()


@lru_cache(maxsize=1)
def _yolo():
    from ultralytics import YOLO

    return YOLO(_settings().yolo_weights)


@lru_cache(maxsize=1)
def _tabular_model():
    s = _settings()
    best_name_file = s.artifacts_root / "best_tabular.txt"
    if not best_name_file.exists():
        raise RuntimeError("No trained model found. Run train-tabular first.")
    name = best_name_file.read_text().strip()
    return joblib.load(s.artifacts_root / f"{name}.joblib"), name


def _feature_row(image: Image.Image) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    model = _yolo()
    class_names = [model.names[i] for i in sorted(model.names)]
    w, h = image.size
    img_area = float(w * h) if w and h else 1.0
    result = model.predict(source=image, conf=0.25, imgsz=640, verbose=False)[0]

    counts = {f"count_{c}": 0 for c in class_names}
    confs = {f"confsum_{c}": 0.0 for c in class_names}
    areas = {f"areafrac_{c}": 0.0 for c in class_names}
    detections: list[dict[str, Any]] = []

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
            detections.append({
                "class": name,
                "confidence": float(cv),
                "bbox_xyxy": [float(v) for v in box],
            })

    row = {**counts, **confs, **areas}
    return pd.DataFrame([row]), detections


app = FastAPI(title="Scene Classification API", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=Prediction)
async def predict(image: UploadFile = File(...)) -> Prediction:
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload must be an image.")

    data = await image.read()
    try:
        pil = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}") from exc

    model, _name = _tabular_model()
    feat_df, detections = _feature_row(pil)

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(feat_df)[0]
        classes = list(model.classes_)
    else:
        pred = model.predict(feat_df)[0]
        classes = list(getattr(model, "classes_", [pred]))
        proba = np.array([1.0 if c == pred else 0.0 for c in classes])

    top_idx = int(np.argmax(proba))
    return Prediction(
        label=classes[top_idx],
        probabilities={c: float(p) for c, p in zip(classes, proba, strict=False)},
        detections=detections,
    )
