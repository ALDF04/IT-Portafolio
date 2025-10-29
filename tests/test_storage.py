from pathlib import Path

import pandas as pd

import storage.cache as cache
import storage.sink as sink


class _FakeRedis(dict):
    def set(self, key, value):  # type: ignore[override]
        self[key] = value

    def get(self, key):  # type: ignore[override]
        return dict.get(self, key)


def test_snapshot_roundtrip(monkeypatch):
    fake = _FakeRedis()
    monkeypatch.setattr(cache, "get_redis", lambda: fake)
    payload = {"value": 42}
    cache.set_snapshot("test", payload)
    stored = cache.get_snapshot("test")
    assert stored == payload


def test_parquet_append(tmp_path, monkeypatch):
    monkeypatch.setattr(sink, "DATA_ROOT", tmp_path)
    df = pd.DataFrame(
        {"close": [1.0, 2.0]},
        index=pd.date_range("2024-01-01", periods=2, freq="1min"),
    )
    path = sink.append_parquet("TEST", df)
    assert Path(path).exists()
    df2 = pd.read_parquet(path)
    assert len(df2) == 2
