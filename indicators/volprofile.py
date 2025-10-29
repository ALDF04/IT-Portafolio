import numpy as np
import pandas as pd


def volume_profile_poc(df: pd.DataFrame, bins: int = 40) -> float | None:
    if df.empty:
        return None
    prices = df["close"].values
    vols = df["volume"].values
    lo, hi = np.nanmin(prices), np.nanmax(prices)
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        return None
    hist, edges = np.histogram(prices, bins=bins, range=(lo, hi), weights=vols)
    idx = int(np.nanargmax(hist))
    poc = (edges[idx] + edges[idx + 1]) / 2.0
    return float(poc)
