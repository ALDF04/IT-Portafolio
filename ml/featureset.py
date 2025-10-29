"""Feature engineering utilities for the AI analyst workflow."""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from indicators.avwap import (
    add_avwap_annual,
    add_avwap_quarterly,
    add_avwap_session,
)
from indicators.features import add_all
from indicators.institutional import apply_institutional_algorithms
from indicators.liquidity_map import liquidity_snapshot
from indicators.signals_pro import (
    add_regime_flags,
    add_rsi,
    add_squeeze_adx,
    detect_squeeze_breakout,
)
from indicators.volprofile import volume_profile_poc


@dataclass(slots=True)
class FeatureSet:
    features: pd.DataFrame
    meta: pd.DataFrame


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex for feature engineering")
    return df.sort_index()


def _atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(length).mean()


def _poc_series(df: pd.DataFrame, lookback: int = 200) -> pd.Series:
    values: list[float | None] = []
    for idx in range(len(df)):
        window = df.iloc[max(0, idx - lookback + 1) : idx + 1]
        poc_value = volume_profile_poc(window)
        values.append(poc_value)
    return pd.Series(values, index=df.index, name="poc")


def _prepare_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = _ensure_datetime_index(df.copy())
    df = add_all(df)
    df = add_avwap_annual(df)
    df = add_avwap_quarterly(df)
    df = add_avwap_session(df)
    df = add_squeeze_adx(df)
    df = add_rsi(df)
    df = add_regime_flags(df)
    df["poc"] = _poc_series(df)
    df["atr"] = _atr(df)
    df["squeeze_flag"] = detect_squeeze_breakout(df).astype(int)
    df, _ = apply_institutional_algorithms(df)
    return df.dropna()


def _bias(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denom = denominator.replace(0, np.nan)
    return (numerator / denom) - 1.0


def _distance_atr(price: pd.Series, reference: pd.Series, atr: pd.Series) -> pd.Series:
    atr_safe = atr.replace(0, np.nan)
    return (price - reference) / atr_safe


def _series_with_default(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column in df:
        return df[column].fillna(default)
    return pd.Series(default, index=df.index)


def build_features(df: pd.DataFrame, lookback_cols: Sequence[str] | None = None) -> FeatureSet:
    enriched = _prepare_indicators(df)
    atr = enriched["atr"]
    close = enriched["close"]

    cvd_series = _series_with_default(enriched, "cvd")

    feature_cols: dict[str, pd.Series] = {
        "bias_ema21": _bias(close, enriched.get("ema_21", close)),
        "bias_ema21w": _bias(close, enriched.get("ema_21w", close)),
        "bias_avwap_y": _bias(close, enriched.get("avwap_y", close)),
        "bias_poc": _bias(close, enriched.get("poc", close)),
        "rvol_z": _series_with_default(enriched, "rvol_z"),
        "cvd_norm": cvd_series.pct_change().fillna(0.0),
        "bb_width": _series_with_default(enriched, "bb_width"),
        "adx": _series_with_default(enriched, "adx"),
        "rsi": _series_with_default(enriched, "rsi"),
        "atr": atr,
        "dist_poc_atr": _distance_atr(close, enriched.get("poc", close), atr),
        "dist_avwap_y_atr": _distance_atr(close, enriched.get("avwap_y", close), atr),
        "dist_session_avwap_atr": _distance_atr(close, enriched.get("avwap_session", close), atr),
        "avwap_session_gap": (close - enriched.get("avwap_session", close)).fillna(0.0),
        "cvd_slope": cvd_series.diff().rolling(5).mean().fillna(0.0),
        "squeeze_flag": _series_with_default(enriched, "squeeze_flag"),
        "momentum_score": _series_with_default(enriched, "momentum_score"),
        "momentum_trend": _series_with_default(enriched, "momentum_trend"),
        "momentum_carry": _series_with_default(enriched, "momentum_carry"),
        "momentum_value": _series_with_default(enriched, "momentum_value"),
        "liquidity_stress": _series_with_default(enriched, "liquidity_stress"),
        "liquidity_spread": _series_with_default(enriched, "liquidity_spread"),
        "liquidity_delta": _series_with_default(enriched, "liquidity_delta"),
        "liquidity_density": _series_with_default(enriched, "liquidity_density"),
        "position_factor": _series_with_default(enriched, "position_factor"),
    }

    for col in enriched.columns:
        if col.startswith("avwap_q"):
            feature_cols[f"bias_{col}"] = _bias(close, enriched[col])
            feature_cols[f"dist_{col}_atr"] = _distance_atr(close, enriched[col], atr)

    if lookback_cols:
        for col in lookback_cols:
            if col in enriched:
                feature_cols[f"{col}_mean_10"] = enriched[col].rolling(10).mean()
                feature_cols[f"{col}_std_10"] = enriched[col].rolling(10).std()

    features = pd.DataFrame(feature_cols).dropna()

    snapshot = liquidity_snapshot()
    liquidity_columns = {
        "risk_score": snapshot.get("risk_score"),
        "qqq_spy_pct": snapshot.get("qqq_spy_pct"),
        "soxx_spy_pct": snapshot.get("soxx_spy_pct"),
        "hyg_tlt_pct": snapshot.get("hyg_tlt_pct"),
        "xlf_xlk_pct": snapshot.get("xlf_xlk_pct"),
        "xle_spy_pct": snapshot.get("xle_spy_pct"),
    }
    for name, value in liquidity_columns.items():
        fill_value = 0.0 if value is None else float(value)
        features[name] = fill_value

    meta = pd.DataFrame(
        {
            "symbol": enriched.get("symbol", pd.Series(index=enriched.index, dtype=object)),
            "timeframe": enriched.get("timeframe", pd.Series(index=enriched.index, dtype=object)),
            "regime": enriched.get("regime", pd.Series(index=enriched.index, dtype=object)),
        }
    ).reindex(features.index)

    dt_index = features.index
    features["hour"] = dt_index.hour
    features["dayofweek"] = dt_index.dayofweek
    features["dist_to_poc"] = features["dist_poc_atr"]

    return FeatureSet(features=features, meta=meta)


def latest_feature_row(df: pd.DataFrame) -> pd.Series:
    feature_set = build_features(df)
    if feature_set.features.empty:
        raise ValueError("Not enough data to compute features")
    return feature_set.features.iloc[-1]


def required_columns() -> Iterable[str]:
    return [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]
