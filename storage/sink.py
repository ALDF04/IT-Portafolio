"""Persistence helpers for streaming data."""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from utils.config import load_settings

DATA_ROOT = Path("data/streams")
DATA_ROOT.mkdir(parents=True, exist_ok=True)


def _timescale_engine() -> Engine | None:
    url = load_settings().TIMESCALE_URL
    if not url:
        return None
    return create_engine(url, future=True, pool_pre_ping=True)


def append_parquet(symbol: str, df: pd.DataFrame) -> Path:
    if df.empty:
        return DATA_ROOT
    symbol_dir = DATA_ROOT / symbol.replace("/", "-")
    symbol_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(df.index, pd.DatetimeIndex):
        date_str = df.index[-1].strftime("%Y%m%d")
    else:
        date_str = "generic"
    path = symbol_dir / f"{date_str}.parquet"
    if path.exists():
        existing = pd.read_parquet(path)
        combined = pd.concat([existing, df]).drop_duplicates()
        combined.to_parquet(path)
    else:
        df.to_parquet(path)
    return path


def write_timescale(table: str, rows: pd.DataFrame) -> None:
    engine = _timescale_engine()
    if engine is None or rows.empty:
        return
    rows.to_sql(table, engine, if_exists="append", index=True, method="multi", chunksize=500)


def persist_snapshot(symbol: str, frames: Iterable[pd.DataFrame]) -> None:
    combined = pd.concat(list(frames)).sort_index()
    append_parquet(symbol, combined)
    write_timescale("ticks", combined.reset_index())

