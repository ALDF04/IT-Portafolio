"""Model training pipeline for the AI analyst module."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover - optional dependency
    XGBClassifier = None  # type: ignore

try:
    from lightgbm import LGBMClassifier
except ImportError:  # pragma: no cover - optional dependency
    LGBMClassifier = None  # type: ignore

MODEL_DIR = Path("models/latest")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def _load_dataset(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_parquet(path)
    y = (df["y_outcome"].astype(float) > 0).astype(int)
    X = df.drop(columns=["y_outcome", "y_entry"], errors="ignore")
    return X, y


def _build_model() -> Any:
    if XGBClassifier is not None:
        return XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
        )
    if LGBMClassifier is not None:
        return LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=-1,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary",
        )
    raise ImportError("Neither xgboost nor lightgbm is installed")


def _build_pipeline() -> Pipeline:
    scaler = StandardScaler()
    model = _build_model()
    return Pipeline([
        ("scaler", scaler),
        ("model", model),
    ])


def _time_series_cv(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
    splitter = TimeSeriesSplit(n_splits=5)
    oof_preds = np.zeros(len(X))
    reports: list[dict[str, Any]] = []

    for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        estimator = clone(pipeline)
        estimator.fit(X_train, y_train)
        proba = estimator.predict_proba(X_test)[:, 1]
        oof_preds[test_idx] = proba
        report = classification_report(y_test, proba > 0.5, output_dict=True, zero_division=0)
        reports.append({"fold": fold, "report": report})

    metrics = {"classification_report": reports}
    metrics["mean_pred"] = float(oof_preds.mean())
    metrics["std_pred"] = float(oof_preds.std())
    return metrics


def train(dataset_path: str = "data/experiments/train.parquet") -> dict[str, Any]:
    X, y = _load_dataset(Path(dataset_path))
    pipeline = _build_pipeline()
    pipeline.fit(X, y)

    calibrator = CalibratedClassifierCV(
        base_estimator=_build_pipeline(),
        method="sigmoid",
        cv=TimeSeriesSplit(n_splits=3),
    )
    calibrator.fit(X, y)

    metrics = _time_series_cv(pipeline, X, y)

    scaler = pipeline.named_steps["scaler"]
    model = pipeline.named_steps["model"]

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_DIR / "model.pkl")
    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")
    joblib.dump(calibrator, MODEL_DIR / "calibrator.pkl")

    with (MODEL_DIR / "metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    with (MODEL_DIR / "features.json").open("w", encoding="utf-8") as fh:
        json.dump({"feature_columns": X.columns.tolist()}, fh, indent=2)

    return metrics


if __name__ == "__main__":  # pragma: no cover
    metrics = train()
    print(json.dumps(metrics, indent=2))
