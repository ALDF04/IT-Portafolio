from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import pandas_ta as ta
except ImportError:  # pragma: no cover - optional dependency
    ta = None



def add_squeeze_adx(df: pd.DataFrame, bb_len: int = 20, adx_len: int = 14) -> pd.DataFrame:
    if ta is None:
        return df
    bb = ta.bbands(df["close"], length=bb_len)
    df = df.join(bb)
    df["bb_width"] = (df["bb_bbh"] - df["bb_bbl"]) / df["bb_bbm"].replace(0, np.nan)
    df["adx"] = ta.adx(df["high"], df["low"], df["close"], length=adx_len)["ADX_" + str(adx_len)]
    return df


def detect_squeeze_breakout(df: pd.DataFrame, pctl: float = 0.2, adx_th: float = 20):
    if ta is None:
        return pd.Series(False, index=df.index)
    thr = df["bb_width"].rolling(200).quantile(pctl)
    return (df["bb_width"] < thr) & (df["adx"] > adx_th) & (df["close"] > df["bb_bbm"])


def add_rsi(df: pd.DataFrame, length: int = 14) -> pd.DataFrame:
    if ta is None:
        df["rsi"] = df["close"].pct_change().fillna(0).cumsum()
        return df
    df["rsi"] = ta.rsi(df["close"], length=length)
    return df


def rsi_divergence_simple(df: pd.DataFrame, lookback: int = 30):
    df["div_bear"] = 0
    df["div_bull"] = 0
    return df


def add_regime_flags(df: pd.DataFrame, vix: float | None = None, dxy: float | None = None):
    regime = "shock" if (vix and vix > 25) else "normal"
    df["regime"] = regime
    return df
