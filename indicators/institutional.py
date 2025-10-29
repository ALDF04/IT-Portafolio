"""Institutional-style algorithmic helpers.

These helpers emulate the high-level logic of popular institutional models
referenced by the team: a momentum factor blend, a hybrid liquidity stress
index, a volatility-adjusted position sizer, and a lightweight correlation
matrix across reference assets.  The goal is to surface structured context for
discretionary decision making without executing orders automatically.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd


def _safe_series(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column in df:
        return df[column].astype(float).fillna(default)
    return pd.Series(default, index=df.index, dtype=float)


def momentum_factor_model(df: pd.DataFrame) -> pd.DataFrame:
    """Blend trend/carry/value style components using EMA structure and RVOL.

    The implementation purposefully remains transparent: trend is derived from
    the slope between medium and long EMAs, carry captures short-term EMA
    momentum, and value gauges distance versus the long-term anchor.  RVOL and
    optional open-interest data modulate the final score.
    """

    if df.empty:
        return df

    result = df.copy()
    close = result["close"].astype(float)
    ema21 = _safe_series(result, "ema_21")
    ema50 = _safe_series(result, "ema_50")
    ema200 = _safe_series(result, "ema_200")

    with np.errstate(divide="ignore", invalid="ignore"):
        trend = ((ema50 - ema200) / ema200.replace(0, np.nan)).fillna(0.0)
        carry = ema21.pct_change(periods=5).rolling(5).mean().fillna(0.0)
        value = ((close - ema200) / ema200.replace(0, np.nan)).fillna(0.0)

    rvol = _safe_series(result, "rvol_z")
    trend = trend * (1 + rvol.clip(-3, 3) / 10.0)

    if "open_interest" in result:
        oi = result["open_interest"].astype(float)
        oi_mean = oi.rolling(20).mean()
        value += ((oi - oi_mean) / oi_mean.replace(0, np.nan)).fillna(0.0)

    score = (0.5 * trend + 0.3 * carry + 0.2 * value).clip(-3, 3)

    result["momentum_trend"] = trend
    result["momentum_carry"] = carry
    result["momentum_value"] = value
    result["momentum_score"] = score
    return result


def hybrid_liquidity_index(df: pd.DataFrame) -> pd.DataFrame:
    """Estimate liquidity stress from spread, delta and volume density inputs."""

    if df.empty:
        return df

    result = df.copy()
    close = result["close"].astype(float)

    if {"bid", "ask"}.issubset(result.columns):
        spread_raw = (result["ask"].astype(float) - result["bid"].astype(float))
    else:
        spread_raw = (result["high"].astype(float) - result["low"].astype(float))

    spread = (spread_raw / close.replace(0, np.nan)).fillna(0.0)
    spread_z = _zscore(spread, window=20)

    volume = result["volume"].astype(float)
    density = (volume / volume.rolling(20).mean().replace(0, np.nan)).fillna(0.0)
    density_z = _zscore(density, window=20)

    delta = _safe_series(result, "delta")
    delta_norm = (delta / volume.replace(0, np.nan)).fillna(0.0)

    score = (-spread_z + 0.5 * delta_norm + 0.5 * density_z).clip(-3, 3)

    result["liquidity_spread"] = spread_z
    result["liquidity_delta"] = delta_norm
    result["liquidity_density"] = density_z
    result["liquidity_stress"] = score
    return result


def volatility_adjusted_position(df: pd.DataFrame, atr_length: int = 14) -> pd.DataFrame:
    """Create a position-sizing proxy using ATR and RVOL as volatility drivers."""

    if df.empty:
        return df

    result = df.copy()
    atr = result["atr"].astype(float) if "atr" in result else _atr(result, atr_length)
    atr_med = atr.rolling(60).median().replace(0, np.nan)
    atr_norm = (atr / atr_med).fillna(1.0)

    rvol = _safe_series(result, "rvol_z").abs() + 1.0
    factor = (1.0 / (atr_norm * rvol)).clip(0.0, 1.0)

    result["position_factor"] = factor
    return result


def ai_correlation_matrix(
    df: pd.DataFrame, benchmarks: Mapping[str, pd.Series] | None = None
) -> dict[str, float | None]:
    """Return the latest rolling correlation versus reference assets."""

    if df.empty or not benchmarks:
        return {}

    close_returns = df["close"].astype(float).pct_change().dropna().rename("asset")
    correlations: dict[str, float | None] = {}
    for label, series in benchmarks.items():
        if series is None or series.empty:
            continue
        series = series.astype(float)
        comparator = series.pct_change().dropna().rename(label)
        joined = pd.concat([close_returns, comparator], axis=1, join="inner").dropna()
        if joined.empty:
            continue
        rolling = joined["asset"].rolling(60).corr(joined[label])
        corr_value = rolling.iloc[-1] if not rolling.empty else joined["asset"].corr(joined[label])
        correlations[label] = float(corr_value) if pd.notna(corr_value) else None
    return correlations


def apply_institutional_algorithms(
    df: pd.DataFrame, benchmarks: Mapping[str, pd.Series] | None = None
) -> tuple[pd.DataFrame, dict[str, float | None]]:
    """Enrich *df* with institutional metrics and return correlation context."""

    enriched = momentum_factor_model(df)
    enriched = hybrid_liquidity_index(enriched)
    enriched = volatility_adjusted_position(enriched)
    correlations = ai_correlation_matrix(enriched, benchmarks)
    return enriched, correlations


def _zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window).mean()
    std = series.rolling(window).std(ddof=0).replace(0, np.nan)
    return ((series - mean) / std).fillna(0.0)


def _atr(df: pd.DataFrame, length: int) -> pd.Series:
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
    return tr.rolling(length).mean().fillna(method="bfill")


__all__ = [
    "apply_institutional_algorithms",
    "ai_correlation_matrix",
    "hybrid_liquidity_index",
    "momentum_factor_model",
    "volatility_adjusted_position",
]

