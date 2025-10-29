"""Session helpers to split cash vs globex flows."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta, timezone

from datafeed import QuoteTick
from orderflow.cvd import CVD
from orderflow.vrvp import VRVP

CASH_START = time(13, 30)  # 09:30 NY in UTC
CASH_END = time(20, 0)  # 16:00 NY in UTC


def is_cash_session(dt_utc: datetime) -> bool:
    if dt_utc.tzinfo is None:
        dt_utc = dt_utc.replace(tzinfo=timezone.utc)
    current_time = dt_utc.timetz()
    return CASH_START <= current_time <= CASH_END


def session_key(dt_utc: datetime) -> str:
    if dt_utc.tzinfo is None:
        dt_utc = dt_utc.replace(tzinfo=timezone.utc)
    session_date = dt_utc.date()
    if is_cash_session(dt_utc):
        return f"cash-{session_date.strftime('%Y%m%d')}"
    return f"globex-{session_date.strftime('%Y%m%d')}"


@dataclass
class SessionState:
    price_step: float = 0.01
    rolling_window: timedelta = timedelta(minutes=30)
    cvd: CVD = field(default_factory=CVD)
    vrvp: VRVP = field(init=False)
    ticks: deque[QuoteTick] = field(default_factory=deque)
    current_key: str | None = None

    def __post_init__(self) -> None:
        self.vrvp = VRVP(price_step=self.price_step)

    def _reset(self, new_key: str) -> None:
        self.current_key = new_key
        self.cvd = CVD()
        self.vrvp.reset()
        self.ticks.clear()

    def update(self, tick: QuoteTick) -> dict[str, float | str | None]:
        timestamp = tick.get("ts")
        if not isinstance(timestamp, datetime):
            raise ValueError("Tick must include a datetime 'ts' key")
        key = session_key(timestamp)
        if key != self.current_key:
            self._reset(key)
        self.cvd.update(tick)
        self.vrvp.update(tick)
        self._append_tick(tick)
        rolling_poc = self._rolling_poc(timestamp)
        return {
            "session_key": key,
            "cvd": self.cvd.value,
            "session_poc": self.vrvp.poc(),
            "rolling_poc": rolling_poc,
        }

    def _append_tick(self, tick: QuoteTick) -> None:
        timestamp = tick["ts"]
        self.ticks.append(tick)
        cutoff = timestamp - self.rolling_window
        while self.ticks and self.ticks[0]["ts"] < cutoff:
            self.ticks.popleft()

    def _rolling_poc(self, now: datetime) -> float | None:
        vrvp = VRVP(price_step=self.price_step)
        for stored_tick in self.ticks:
            vrvp.update(stored_tick)
        return vrvp.poc()

