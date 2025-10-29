"""Cumulative volume delta helper tracking aggressive flows."""
from __future__ import annotations

from dataclasses import dataclass

from datafeed import QuoteTick


@dataclass
class CVD:
    value: float = 0.0

    def update(self, tick: QuoteTick) -> float:
        side = tick.get("side")
        size = float(tick.get("size", 0.0) or 0.0)
        if side == "ASK":
            self.value += size
        elif side == "BID":
            self.value -= size
        return self.value

