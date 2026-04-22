# Scene Classification — YOLO object features vs. transfer-learned CNN

An honest redo of *Scene Classification with Simple Machine Learning and Convolutional Neural Network*
(Yosboon, 2022) with a credible dataset size, modern tooling, reproducible pipeline,
experiment tracking, and a deployable inference API.

**Question.** Can a cheap, interpretable tabular classifier, trained on object-detection counts
from a pretrained YOLO model, compete with a transfer-learned CNN on indoor scene
classification?

**Why it's a portfolio piece.** The pipeline is end-to-end — data download, feature
engineering with a pretrained model, classical ML *and* deep learning, MLflow-tracked
experiments, SHAP interpretability, a FastAPI service, Docker, CI — each piece small on
purpose so the whole is easy to read.

## Pipeline

```
raw images ──► YOLOv8 detector ──► per-COCO-class counts / confsum / areafrac
                                              │
                                              ▼
                      ┌────────────────────────────────────────┐
                      │ DT  KNN  NB  SVM  LogReg  XGBoost      │  ◄── MLflow
                      └────────────────────────────────────────┘
                                              │
raw images ──► ResNet50 (ImageNet, fine-tuned) ┘
                                              ▼
                         confusion matrices + comparison chart
                                              │
                                              ▼
                             FastAPI  /predict  (Docker)
```

## Dataset

**MIT Indoor Scenes 67** (Quattoni & Torralba, 2009). By default we subset to the four
classes used in the original paper — `bedroom`, `bathroom`, `livingroom`, `kitchen` — so
the comparison tracks. The downloader can take any subset of the 67 classes via
`--classes`.

## Quickstart

```bash
make setup                # uv venv + editable install
scenes prepare-data       # download MIT Indoor67 + stratified 70/15/15 split
scenes extract-features   # YOLOv8 -> data/processed/features.csv
scenes train-tabular      # DT/KNN/NB/SVM/LR/XGB + MLflow
scenes train-cnn          # ResNet50 transfer learning + MLflow
scenes evaluate           # figures + reports/summary.md
make serve                # FastAPI at http://localhost:8000/docs
```

## Results

Not yet trained — this README gets filled in once `scenes evaluate` runs. Do not paste
numbers here until you've actually produced them; the summary writer emits
`reports/summary.md` and figures under `reports/figures/`.

| Model | CV acc | Val acc | Test acc | Macro-F1 | Train seconds |
|---|---|---|---|---|---|
| Decision Tree | — | — | — | — | — |
| KNN | — | — | — | — | — |
| Naive Bayes | — | — | — | — | — |
| SVM (RBF) | — | — | — | — | — |
| Logistic Regression | — | — | — | — | — |
| XGBoost | — | — | — | — | — |
| ResNet50 (transfer) | n/a | — | — | — | — |

## What's deliberately different from the paper

- **Dataset.** MIT Indoor67 instead of 400 hand-picked internet images.
- **Feature vector.** Per-class `count + confsum + areafrac` (3 × 80 dims on COCO) instead of raw 41 counts — richer signal, still interpretable.
- **Detector.** YOLOv8 (Ultralytics) instead of YOLOv3.
- **CNN baseline.** ResNet50 transfer learning (ImageNet weights, head + last block unfrozen, early stopping) instead of Inception-v3 trained from scratch with batch size 1 — a fair deep-learning comparison.
- **Models.** Paper baselines retained; Logistic Regression and XGBoost added.
- **Evaluation.** Stratified 5-fold CV, held-out val + test, accuracy *and* macro-F1, per-class confusion matrices, training-time / inference-latency comparison.
- **Tracking.** MLflow per run, artifacts checked in under `mlruns/` locally.

## Limitations

- COCO object vocabulary is the prior: this approach is only as good as YOLO's coverage of a scene type. Works for indoor rooms; would not work for most outdoor/natural categories.
- Object counts miss layout and spatial arrangement, so the tabular path has a ceiling the CNN can surpass given enough data.
- Single-detector feature set — no ensembling of detectors or scene-level embeddings.

## Repo layout

```
src/scene_classification/
  config.py                 settings from env vars
  cli.py                    `scenes` Typer CLI
  data/download.py          MIT Indoor67 download + stratified split
  features/extract.py       YOLOv8 feature extraction
  models/train_tabular.py   DT / KNN / NB / SVM / LR / XGB + MLflow
  models/train_cnn.py       ResNet50 transfer learning + MLflow
  evaluate.py               confusion matrices + summary.md
  serve/api.py              FastAPI /health + /predict
tests/                      starter unit tests
.github/workflows/ci.yml    ruff + pytest on push/PR
Dockerfile                  slim image that serves the API
```

## License

MIT — see [LICENSE](LICENSE).
