import pandas as pd
from redis import Redis

from .config import load_settings


def get_redis() -> Redis | None:
    url = load_settings().REDIS_URL
    return Redis.from_url(url) if url else None


def set_df(key: str, df: pd.DataFrame) -> None:
    client = get_redis()
    if client is None:
        return
    client.set(key, df.to_json(orient="split"))


def get_df(key: str) -> pd.DataFrame | None:
    client = get_redis()
    if client is None:
        return None
    raw = client.get(key)
    return pd.read_json(raw, orient="split") if raw else None
