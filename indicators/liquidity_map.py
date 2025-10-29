"""Helpers to compute intermarket liquidity signals for the dashboard and ML features."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd
import yfinance as yf


@dataclass(slots=True)
class SeriesWithMeta:
    symbol: str
    series: pd.Series


def fetch_close(symbol: str, period: str = "6mo", interval: str = "1d") -> SeriesWithMeta:
    """Download close prices for a symbol using yfinance."""

    data = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=False)
    if data.empty:
        raise ValueError(f"No data for {symbol}")
    close = data["Close"].rename("close").dropna()
    close.index = pd.to_datetime(close.index)
    return SeriesWithMeta(symbol=symbol, series=close)


def pct_change(series: pd.Series, periods: int = 1) -> float | None:
    if series.empty or len(series) <= periods:
        return None
    latest = series.iloc[-1]
    prev = series.iloc[-(periods + 1)]
    if prev == 0:
        return None
    return float((latest - prev) / prev)


def sign_state(value: float | None, invert: bool = False, threshold: float = 0.0) -> int | None:
    if value is None:
        return None
    signed = -value if invert else value
    if signed > threshold:
        return 1
    if signed < -threshold:
        return -1
    return 0


def weekly_proxy(series: pd.Series) -> pd.Series:
    if series.empty:
        return series
    weekly = series.resample("W").last().dropna()
    return weekly


def ratio_series(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    df = pd.concat([numerator, denominator], axis=1, join="inner").dropna()
    if df.empty:
        return pd.Series(dtype=float)
    ratio = df.iloc[:, 0] / df.iloc[:, 1]
    ratio.name = "ratio"
    return ratio


def risk_score(
    dxy_pct: float | None,
    vix_pct: float | None,
    tnx_pct: float | None,
    tlt_pct: float | None,
) -> float | None:
    weights = (
        (dxy_pct, -0.30),
        (vix_pct, -0.25),
        (tnx_pct, -0.25),
        (tlt_pct, 0.20),
    )
    total = 0.0
    weight_sum = 0.0
    for value, weight in weights:
        if value is None:
            continue
        total += value * weight
        weight_sum += abs(weight)
    if weight_sum == 0:
        return None
    return total / weight_sum


def score_opportunity(changes: Sequence[float | None], weights: Sequence[float]) -> float | None:
    if not changes or not weights or len(changes) != len(weights):
        return None
    total = 0.0
    weight_sum = 0.0
    for value, weight in zip(changes, weights, strict=False):
        if value is None:
            continue
        total += value * weight
        weight_sum += weight
    if weight_sum == 0:
        return None
    return total / weight_sum


def _safe_pct(series: pd.Series) -> float | None:
    try:
        return pct_change(series)
    except Exception:  # pragma: no cover - defensive
        return None


@pd.api.extensions.register_dataframe_accessor("liquidity")
class LiquidityAccessor:
    """Convenience accessors to compute ratios on DataFrames of close prices."""

    def __init__(self, pandas_obj: pd.DataFrame) -> None:
        self._obj = pandas_obj

    def pct(self, symbol: str) -> float | None:
        series = self._obj.get(symbol)
        if series is None:
            return None
        return _safe_pct(series.dropna())


def liquidity_snapshot() -> dict[str, float | None]:
    symbols = {
        "dxy": "DX-Y.NYB",
        "vix": "^VIX",
        "tnx": "^TNX",
        "tlt": "TLT",
        "qqq": "QQQ",
        "spy": "SPY",
        "soxx": "SOXX",
        "hyg": "HYG",
        "xlf": "XLF",
        "xlk": "XLK",
        "xle": "XLE",
        "xlp": "XLP",
        "xli": "XLI",
        "xlv": "XLV",
    }
    closes: dict[str, pd.Series] = {}
    for key, ticker in symbols.items():
        try:
            closes[key] = fetch_close(ticker).series
        except Exception:
            closes[key] = pd.Series(dtype=float)

    ratios = {
        "qqq_spy": ratio_series(
            closes.get("qqq", pd.Series(dtype=float)),
            closes.get("spy", pd.Series(dtype=float)),
        ),
        "soxx_spy": ratio_series(
            closes.get("soxx", pd.Series(dtype=float)),
            closes.get("spy", pd.Series(dtype=float)),
        ),
        "hyg_tlt": ratio_series(
            closes.get("hyg", pd.Series(dtype=float)),
            closes.get("tlt", pd.Series(dtype=float)),
        ),
        "xlf_xlk": ratio_series(
            closes.get("xlf", pd.Series(dtype=float)),
            closes.get("xlk", pd.Series(dtype=float)),
        ),
        "xle_spy": ratio_series(
            closes.get("xle", pd.Series(dtype=float)),
            closes.get("spy", pd.Series(dtype=float)),
        ),
    }

    dxy_pct = _safe_pct(closes.get("dxy", pd.Series(dtype=float)))
    vix_pct = _safe_pct(closes.get("vix", pd.Series(dtype=float)))
    tnx_pct = _safe_pct(closes.get("tnx", pd.Series(dtype=float)))
    tlt_pct = _safe_pct(closes.get("tlt", pd.Series(dtype=float)))

    snapshot = {
        "risk_score": risk_score(dxy_pct, vix_pct, tnx_pct, tlt_pct),
        "dxy_pct": dxy_pct,
        "vix_pct": vix_pct,
        "tnx_pct": tnx_pct,
        "tlt_pct": tlt_pct,
    }
    for key, ratio in ratios.items():
        snapshot[f"{key}_pct"] = _safe_pct(ratio)
    for sector in ["xlk", "soxx", "xle", "xlf", "xlp", "xli", "xlv"]:
        snapshot[f"{sector}_pct"] = _safe_pct(closes.get(sector, pd.Series(dtype=float)))
    return snapshot
