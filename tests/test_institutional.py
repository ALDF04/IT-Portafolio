import pandas as pd

from indicators.institutional import (
    ai_correlation_matrix,
    apply_institutional_algorithms,
    hybrid_liquidity_index,
    momentum_factor_model,
    volatility_adjusted_position,
)


def _sample_df(rows: int = 120) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=rows, freq="D")
    data = {
        "open": pd.Series(range(rows), index=idx, dtype=float) + 100.0,
        "high": pd.Series(range(rows), index=idx, dtype=float) + 101.0,
        "low": pd.Series(range(rows), index=idx, dtype=float) + 99.0,
        "close": pd.Series(range(rows), index=idx, dtype=float) + 100.5,
        "volume": pd.Series(
            [1_000 + 5 * (i % 10) for i in range(rows)], index=idx, dtype=float
        ),
    }
    df = pd.DataFrame(data)
    df["ema_21"] = df["close"].ewm(span=21, adjust=False).mean()
    df["ema_50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["ema_200"] = df["close"].ewm(span=200, adjust=False).mean()
    df["rvol_z"] = (df["volume"] - df["volume"].rolling(20).mean()) / df["volume"].rolling(20).std()
    df["delta"] = df["volume"].diff().fillna(0.0)
    df["atr"] = (df["high"] - df["low"]).rolling(14).mean()
    return df.bfill()


def test_momentum_factor_model_outputs_columns():
    df = _sample_df()
    enriched = momentum_factor_model(df)
    for column in ["momentum_score", "momentum_trend", "momentum_carry", "momentum_value"]:
        assert column in enriched.columns


def test_hybrid_liquidity_index_outputs_columns():
    df = _sample_df()
    enriched = hybrid_liquidity_index(df)
    for column in ["liquidity_stress", "liquidity_spread", "liquidity_delta", "liquidity_density"]:
        assert column in enriched.columns


def test_volatility_adjusted_position_column():
    df = _sample_df()
    enriched = volatility_adjusted_position(df)
    assert "position_factor" in enriched.columns


def test_apply_institutional_algorithms_with_correlations():
    df = _sample_df()
    benchmarks = {"SPX": df["close"].rename("SPX").shift(1)}
    enriched, corr = apply_institutional_algorithms(df, benchmarks)
    assert "momentum_score" in enriched.columns
    assert corr


def test_ai_correlation_matrix_handles_empty():
    df = _sample_df()
    assert ai_correlation_matrix(df, {}) == {}
