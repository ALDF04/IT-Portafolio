"""Explanation utilities using SHAP for the AI analyst."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

try:
    import shap
except ImportError:  # pragma: no cover - optional dependency
    shap = None  # type: ignore

MODEL_DIR = Path("models/latest")


def _load_model_artifacts() -> dict[str, Any]:
    model = joblib.load(MODEL_DIR / "model.pkl")
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    calibrator = joblib.load(MODEL_DIR / "calibrator.pkl")
    return {"model": model, "scaler": scaler, "calibrator": calibrator}


def explain_global(X: pd.DataFrame, max_display: int = 10) -> dict[str, Any]:
    if shap is None:
        return {"message": "SHAP not installed", "importances": X.mean().to_dict()}
    artifacts = _load_model_artifacts()
    model = artifacts["model"]
    scaler = artifacts["scaler"]
    X_scaled = scaler.transform(X)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_scaled)
    if isinstance(shap_values, list):
        shap_array = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    else:
        shap_array = shap_values
    mean_importance = np.mean(np.abs(shap_array), axis=0)
    feature_order = np.argsort(mean_importance)[::-1][:max_display]
    ordered = [
        (X.columns[i], float(mean_importance[i]))
        for i in feature_order
    ]
    return {"importances": ordered}


def explain_sample(sample: pd.Series) -> dict[str, Any]:
    if shap is None:
        return {"message": "SHAP not installed"}
    artifacts = _load_model_artifacts()
    model = artifacts["model"]
    scaler = artifacts["scaler"]
    X_scaled = scaler.transform(sample.to_frame().T)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_scaled)
    if isinstance(shap_values, list):
        shap_array = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    else:
        shap_array = shap_values
    contributions = dict(zip(sample.index, shap_array[0].tolist()))
    return {"values": contributions}
