from datetime import datetime, timezone

from datafeed import QuoteTick
from orderflow.cvd import CVD
from orderflow.vrvp import VRVP


def _tick(price: float, size: float, side: str) -> QuoteTick:
    return QuoteTick(
        ts=datetime.now(timezone.utc),
        symbol="TEST",
        price=price,
        size=size,
        side=side,
    )


def test_cvd_updates_with_aggression():
    cvd = CVD()
    cvd.update(_tick(100.0, 1.0, "ASK"))
    assert cvd.value == 1.0
    cvd.update(_tick(99.5, 2.0, "BID"))
    assert cvd.value == -1.0


def test_vrvp_tracks_poc():
    vrvp = VRVP(price_step=0.5)
    vrvp.update(_tick(100.0, 1.0, "ASK"))
    vrvp.update(_tick(100.1, 3.0, "ASK"))
    vrvp.update(_tick(101.0, 2.0, "BID"))
    assert round(vrvp.poc(), 1) == 100.0
