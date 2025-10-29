from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    import pandas_ta as ta
except ImportError:  # pragma: no cover - optional dependency
    ta = None


@dataclass
class EMAState:
    value: float
    alpha: float

    @classmethod
    def initialise(cls, first_value: float, length: int) -> EMAState:
        alpha = 2 / (length + 1)
        return cls(value=first_value, alpha=alpha)

    def update(self, price: float) -> float:
        self.value = self.alpha * price + (1 - self.alpha) * self.value
        return self.value


@dataclass
class RVOLState:
    lookback: int
    values: list[float]

    @classmethod
    def initialise(cls, lookback: int = 20) -> RVOLState:
        return cls(lookback=lookback, values=[])

    def update(self, volume: float) -> float:
        self.values.append(volume)
        if len(self.values) > self.lookback:
            self.values = self.values[-self.lookback :]
        series = pd.Series(self.values)
        mean = series.mean()
        std = series.std(ddof=0)
        if std == 0:
            return 0.0
        return (series.iloc[-1] - mean) / std


def _ema(series: pd.Series, length: int) -> pd.Series:
    if ta is not None:
        return ta.ema(series, length=length)
    return series.ewm(span=length, adjust=False).mean()


def add_emas(df: pd.DataFrame) -> pd.DataFrame:
    for length in [9, 21, 50, 200]:
        df[f"ema_{length}"] = _ema(df["close"], length)
    return df


def add_rvol(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    vol = df["volume"].copy()
    mean = vol.rolling(lookback).mean()
    std = vol.rolling(lookback).std(ddof=0)
    df["rvol_z"] = (vol - mean) / (std.replace(0, np.nan))
    return df


def add_cvd_simple(df: pd.DataFrame) -> pd.DataFrame:
    ret = df["close"].pct_change().fillna(0.0)
    up_vol = np.where(ret >= 0, df["volume"], 0.0)
    down_vol = np.where(ret < 0, df["volume"], 0.0)
    df["delta"] = up_vol - down_vol
    df["cvd"] = pd.Series(df["delta"], index=df.index).cumsum()
    return df


def add_ema21_weekly_on_daily(df_daily: pd.DataFrame) -> pd.DataFrame:
    df_w = df_daily["close"].resample("W").last()
    ema21w = _ema(df_w, 21).rename("ema_21w")
    df_daily["ema_21w"] = ema21w.reindex(df_daily.index, method="ffill")
    return df_daily


def add_all(df: pd.DataFrame) -> pd.DataFrame:
    df = add_emas(df)
    df = add_rvol(df)
    df = add_cvd_simple(df)
    try:
        df = add_ema21_weekly_on_daily(df)
    except Exception:  # pragma: no cover - resample guard
        pass
    return df.dropna()
