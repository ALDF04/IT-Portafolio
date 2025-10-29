import pandas as pd

from indicators.features import add_all


def test_add_all_shape():
    df = pd.DataFrame(
        {
            "open": [1, 2, 3, 4, 5],
            "high": [1, 2, 3, 4, 5],
            "low": [1, 2, 3, 4, 5],
            "close": [1, 2, 3, 4, 5],
            "volume": [10, 10, 10, 10, 10],
        }
    )
    df.index = pd.date_range("2024-01-01", periods=5, freq="D")
    out = add_all(df.copy())
    assert "ema_21" in out.columns
    assert "rvol_z" in out.columns
    assert "cvd" in out.columns
