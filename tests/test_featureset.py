import pandas as pd

from ml.featureset import FeatureSet, build_features


def test_build_features_contains_expected_columns():
    values = list(range(1, 301))
    df = pd.DataFrame(
        {
            "open": values,
            "high": values,
            "low": values,
            "close": values,
            "volume": list(range(100, 100 + len(values))),
        }
    )
    df.index = pd.date_range("2023-01-01", periods=len(df), freq="D")

    feature_set = build_features(df)
    assert isinstance(feature_set, FeatureSet)
    expected = {
        "bias_ema21",
        "rvol_z",
        "atr",
        "risk_score",
        "qqq_spy_pct",
        "momentum_score",
        "liquidity_stress",
        "position_factor",
    }
    assert expected.issubset(feature_set.features.columns)
    assert (feature_set.features["risk_score"] == feature_set.features["risk_score"].iloc[0]).all()
