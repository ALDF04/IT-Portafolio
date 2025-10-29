from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def _avwap_series(df: pd.DataFrame, start_idx: pd.Timestamp) -> pd.Series:
    mask = df.index >= start_idx
    px = df.loc[mask, "close"]
    vol = df.loc[mask, "volume"]
    num = (px * vol).cumsum()
    den = vol.cumsum().replace(0, np.nan)
    out = (num / den).reindex(df.index, method="ffill")
    return out


def _aligned_timestamp(index: pd.DatetimeIndex, target: pd.Timestamp) -> pd.Timestamp:
    position = index.get_indexer([target], method="bfill")
    loc = position[0] if len(position) else -1
    if loc == -1:
        return index[0]
    return index[loc]


def add_avwap_annual(df: pd.DataFrame) -> pd.DataFrame:
    start = pd.Timestamp(year=df.index[0].year, month=1, day=1, tz=df.index.tz)
    start = _aligned_timestamp(df.index, start)
    df["avwap_y"] = _avwap_series(df, start)
    return df


def add_avwap_quarterly(df: pd.DataFrame) -> pd.DataFrame:
    qmonths = [1, 4, 7, 10]
    for qm in qmonths:
        start = pd.Timestamp(year=df.index[0].year, month=qm, day=1, tz=df.index.tz)
        start = _aligned_timestamp(df.index, start)
        df[f"avwap_q{qm}"] = _avwap_series(df, start)
    return df


def add_avwap_session(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        return df
    session_col = []
    grouped = df.groupby(df.index.date)
    for _, daily in grouped:
        start = daily.index[0]
        series = _avwap_series(daily, start)
        session_col.append(series)
    if session_col:
        df["avwap_session"] = pd.concat(session_col).sort_index()
    return df


def add_avwap_event(
    df: pd.DataFrame, event_ts: pd.Timestamp, column: str = "avwap_event"
) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        return df
    aligned = _aligned_timestamp(df.index, event_ts)
    df[column] = _avwap_series(df, aligned)
    return df


@dataclass
class AVWAPState:
    anchor_price: float
    anchor_volume: float

    @classmethod
    def from_bar(cls, price: float, volume: float) -> AVWAPState:
        return cls(anchor_price=price * volume, anchor_volume=volume)

    def update(self, price: float, volume: float) -> float:
        self.anchor_price += price * volume
        self.anchor_volume += volume
        if self.anchor_volume == 0:
            return price
        return self.anchor_price / self.anchor_volume
