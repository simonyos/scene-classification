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

MIT Indoor67 subset: `bathroom`, `bedroom`, `kitchen`, `livingroom` — 2,299 images total,
stratified 70/15/15 split. Tabular models see YOLOv8n object features (80 COCO classes ×
{count, confsum, areafrac} = 240 dims). CNN sees raw images at 224×224.

| Model | Val acc | **Test acc** | Macro-F1 | Train seconds |
|---|---|---|---|---|
| **ResNet50 (transfer)** | **0.892** | **0.845** | **0.838** | — (MPS, 15 epochs w/ early stop) |
| SVM (linear, C=0.5) | 0.810 | 0.788 | 0.763 | 1.9 |
| Logistic Regression (C=0.3) | 0.793 | 0.756 | 0.747 | 0.2 |
| Gradient Boosting (depth=3, 200 trees) | 0.793 | 0.751 | 0.733 | 18.7 |
| Decision Tree (max_depth=10) | 0.767 | 0.716 | 0.702 | 1.8 |
| KNN (k=9) | 0.717 | 0.696 | 0.678 | 0.2 |
| Naive Bayes (Gaussian) | 0.356 | 0.338 | 0.326 | 0.1 |

Full summary with per-run params: [`reports/summary.md`](reports/summary.md).
Confusion matrices per model: [`reports/figures/`](reports/figures/).

**Takeaway.** Opposite of the paper's conclusion — with ~5× more data and a properly
trained CNN (transfer learning, not from-scratch Inception-v3 with batch size 1),
ResNet50 beats the best tabular baseline by **~5.7 points** on test accuracy.
The tabular YOLO-feature pipeline still pulls ~79% with SVM in under 2 s of training and
offers clean interpretability (object-count features) — useful when latency, size, or
explainability matter more than the last few points of accuracy.

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
