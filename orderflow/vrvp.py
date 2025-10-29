"""Volume range volume profile (VRVP) helper."""
from __future__ import annotations

from dataclasses import dataclass, field

from datafeed import QuoteTick


@dataclass
class VRVP:
    price_step: float = 0.01
    bins: dict[float, float] = field(default_factory=dict)

    def _bin(self, price: float) -> float:
        if self.price_step <= 0:
            return price
        return round(price / self.price_step) * self.price_step

    def update(self, tick: QuoteTick) -> None:
        price = float(tick.get("price", 0.0) or 0.0)
        size = float(tick.get("size", 0.0) or 0.0)
        if price == 0 or size == 0:
            return
        bucket = self._bin(price)
        self.bins[bucket] = self.bins.get(bucket, 0.0) + size

    def poc(self) -> float | None:
        if not self.bins:
            return None
        return max(self.bins.items(), key=lambda item: item[1])[0]

    def reset(self) -> None:
        self.bins.clear()

