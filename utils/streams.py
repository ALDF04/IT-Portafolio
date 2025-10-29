import asyncio
from datetime import datetime, timezone

import pandas as pd


class RingBuffer:
    def __init__(self, maxlen: int = 5000):
        self.buf: list[dict] = []
        self.maxlen = maxlen

    def push(self, item: dict) -> None:
        self.buf.append(item)
        if len(self.buf) > self.maxlen:
            self.buf = self.buf[-self.maxlen :]

    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.buf)


async def stream_trades_placeholder(symbol: str, on_trade):
    """Placeholder streamer that simulates trades every 250ms."""
    while True:
        await asyncio.sleep(0.25)
        trade = {
            "ts": datetime.now(timezone.utc),
            "price": None,
            "qty": None,
            "side": None,
            "symbol": symbol,
        }
        await on_trade(trade)
