"""Label generation helpers for supervised training."""
from __future__ import annotations

import pandas as pd

from ml.featureset import build_features


def _confirmation_score(features: pd.DataFrame) -> pd.Series:
    signals = pd.DataFrame(index=features.index)
    signals["ema21"] = features["bias_ema21"] > 0
    signals["ema21w"] = features["bias_ema21w"] > 0
    signals["avwap_y"] = features["bias_avwap_y"] > 0
    signals["rvol"] = features["rvol_z"] > 0
    signals["adx"] = features["adx"] > 20
    return signals.sum(axis=1)


def make_labels(df: pd.DataFrame, horizon: int = 20, rr: float = 1.0) -> pd.DataFrame:
    """Create entry and outcome labels from OHLCV data."""
    feature_set = build_features(df)
    features = feature_set.features
    enriched = df.loc[features.index].copy()
    enriched["atr"] = features["atr"]

    confirmations = _confirmation_score(features)
    y_entry = (confirmations >= 4).astype(int)

    close = enriched["close"].to_numpy()
    high = enriched["high"].to_numpy()
    low = enriched["low"].to_numpy()
    atr = enriched["atr"].to_numpy()

    outcome: list[int] = []
    indices = features.index.tolist()
    for i, _idx in enumerate(indices):
        atr_val = atr[i]
        if not pd.notna(atr_val):
            outcome.append(0)
            continue
        entry = close[i]
        target = entry + rr * atr_val
        stop = entry - rr * atr_val
        horizon_end = min(i + horizon, len(close) - 1)
        future_range = range(i + 1, horizon_end + 1)
        result = 0
        for j in future_range:
            if pd.notna(high[j]) and high[j] >= target:
                result = 1
                break
            if pd.notna(low[j]) and low[j] <= stop:
                result = -1
                break
        outcome.append(result)

    labels = pd.DataFrame(
        {
            "y_entry": y_entry.loc[features.index],
            "y_outcome": pd.Series(outcome, index=features.index),
        }
    )
    return labels
