"""Redis snapshot helpers for near real-time metrics."""
from __future__ import annotations

import json
from typing import Any

import pandas as pd
from redis import Redis

from utils.config import load_settings


def get_redis() -> Redis | None:
    url = load_settings().REDIS_URL
    if not url:
        return None
    return Redis.from_url(url)


def set_snapshot(key: str, payload: dict[str, Any]) -> None:
    client = get_redis()
    if client is None:
        return
    client.set(key, json.dumps(payload, default=str))


def get_snapshot(key: str) -> dict[str, Any] | None:
    client = get_redis()
    if client is None:
        return None
    raw = client.get(key)
    if raw is None:
        return None
    return json.loads(raw)


def set_df(key: str, df: pd.DataFrame) -> None:
    client = get_redis()
    if client is None:
        return
    client.set(key, df.to_json(orient="split", date_format="iso"))


def get_df(key: str) -> pd.DataFrame | None:
    client = get_redis()
    if client is None:
        return None
    raw = client.get(key)
    if not raw:
        return None
    return pd.read_json(raw, orient="split")

