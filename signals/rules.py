import pandas as pd


def signal_ema_reclaim_rvol(df: pd.DataFrame, rvol_min: float = 1.3) -> pd.DataFrame:
    cond = (df["close"] > df["ema_21"]) & (df["ema_9"] > df["ema_21"]) & (df["rvol_z"] > rvol_min)
    df["long_signal"] = cond.astype(int)
    return df

def signal_alignment_21w(df: pd.DataFrame, rvol_min: float = 1.3) -> pd.DataFrame:
    base = signal_ema_reclaim_rvol(df, rvol_min=rvol_min)
    df["aplus_signal"] = (
        (base["long_signal"] == 1)
        & (df.get("ema_21w", df["ema_21"]) < df["close"])
    ).astype(int)
    return df
