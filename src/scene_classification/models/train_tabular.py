"""
Tabular classifiers over YOLO object features.

Models: Decision Tree, KNN, Naive Bayes, SVM (paper baselines) + Logistic Regression and
XGBoost (stronger modern baselines). Experiments tracked to MLflow.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from scene_classification.config import Settings


@dataclass
class TabularResult:
    name: str
    best_params: dict
    cv_accuracy: float
    val_accuracy: float
    test_accuracy: float
    test_macro_f1: float
    train_seconds: float
    model_path: Path


def _features_and_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    feature_cols = [c for c in df.columns if c.startswith(("count_", "confsum_", "areafrac_"))]
    return df[feature_cols], df["label"]


def _model_specs() -> dict[str, tuple[Pipeline, dict]]:
    return {
        "decision_tree": (
            Pipeline([("clf", DecisionTreeClassifier(random_state=0))]),
            {"clf__max_depth": [6, 10, 14, 17, 20, None]},
        ),
        "knn": (
            Pipeline([("scaler", StandardScaler()), ("clf", KNeighborsClassifier())]),
            {"clf__n_neighbors": [3, 4, 5, 7, 9]},
        ),
        "naive_bayes": (
            Pipeline([("clf", GaussianNB())]),
            {},
        ),
        "svm": (
            Pipeline([("scaler", StandardScaler()), ("clf", SVC(probability=True))]),
            {"clf__C": [0.5, 1.0, 4.0], "clf__kernel": ["rbf", "linear"]},
        ),
        "logistic_regression": (
            Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=2000, multi_class="auto")),
            ]),
            {"clf__C": [0.3, 1.0, 3.0]},
        ),
        "xgboost": (
            Pipeline([
                (
                    "clf",
                    XGBClassifier(
                        tree_method="hist",
                        eval_metric="mlogloss",
                        random_state=0,
                        n_jobs=-1,
                    ),
                )
            ]),
            {
                "clf__max_depth": [4, 6, 8],
                "clf__n_estimators": [200, 400],
                "clf__learning_rate": [0.05, 0.1],
            },
        ),
    }


def _fit_one(
    name: str,
    pipeline: Pipeline,
    grid: dict,
    splits: dict[str, tuple[pd.DataFrame, pd.Series]],
    artifacts_dir: Path,
    seed: int,
) -> TabularResult:
    X_train, y_train = splits["train"]
    X_val, y_val = splits["val"]
    X_test, y_test = splits["test"]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    search = GridSearchCV(pipeline, grid or {"clf__random_state": [seed]}, cv=cv, n_jobs=-1)

    t0 = time.perf_counter()
    search.fit(X_train, y_train)
    train_seconds = time.perf_counter() - t0

    best = search.best_estimator_
    val_pred = best.predict(X_val)
    test_pred = best.predict(X_test)

    result = TabularResult(
        name=name,
        best_params=search.best_params_,
        cv_accuracy=float(search.best_score_),
        val_accuracy=float(accuracy_score(y_val, val_pred)),
        test_accuracy=float(accuracy_score(y_test, test_pred)),
        test_macro_f1=float(f1_score(y_test, test_pred, average="macro")),
        train_seconds=train_seconds,
        model_path=artifacts_dir / f"{name}.joblib",
    )

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best, result.model_path)

    report = classification_report(y_test, test_pred, output_dict=True)
    (artifacts_dir / f"{name}_classification_report.json").write_text(json.dumps(report, indent=2))

    return result


def run(settings: Settings) -> list[TabularResult]:
    df = pd.read_csv(settings.features_csv)
    X, y = _features_and_labels(df)
    splits = {s: (X[df["split"] == s], y[df["split"] == s]) for s in ("train", "val", "test")}

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment("scene-classification/tabular")

    results: list[TabularResult] = []
    for name, (pipeline, grid) in _model_specs().items():
        with mlflow.start_run(run_name=name):
            result = _fit_one(name, pipeline, grid, splits, settings.artifacts_root, settings.seed)
            mlflow.log_params({"model": name, **{k: str(v) for k, v in result.best_params.items()}})
            mlflow.log_metrics({
                "cv_accuracy": result.cv_accuracy,
                "val_accuracy": result.val_accuracy,
                "test_accuracy": result.test_accuracy,
                "test_macro_f1": result.test_macro_f1,
                "train_seconds": result.train_seconds,
            })
            mlflow.log_artifact(str(result.model_path))
            results.append(result)

    summary = pd.DataFrame([r.__dict__ for r in results]).drop(columns=["model_path"])
    summary["model_path"] = [str(r.model_path) for r in results]
    summary.to_csv(settings.artifacts_root / "tabular_summary.csv", index=False)

    best = max(results, key=lambda r: r.test_accuracy)
    (settings.artifacts_root / "best_tabular.txt").write_text(best.name)
    np.save(settings.artifacts_root / "classes.npy", np.array(sorted(y.unique())))
    return results
