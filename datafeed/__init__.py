"""Datafeed factory helpers for equities and crypto providers."""
from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Iterable
from typing import Protocol

from utils.config import load_settings

LOGGER = logging.getLogger(__name__)


class QuoteTick(dict):
    """Typed alias for quote/trade tick dictionaries."""


class BaseFeed(Protocol):
    async def subscribe_quotes(self, symbols: Iterable[str]) -> AsyncIterator[QuoteTick]:
        ...

    async def subscribe_trades(self, symbols: Iterable[str]) -> AsyncIterator[QuoteTick]:
        ...


async def _simulated_stream(symbols: Iterable[str], kind: str) -> AsyncIterator[QuoteTick]:
    """Yield placeholder ticks while a real provider is not configured."""

    import random
    from datetime import datetime, timezone

    LOGGER.warning("Using simulated %s feed for symbols %s", kind, ", ".join(symbols))
    while True:
        await asyncio.sleep(1.0)
        for symbol in symbols:
            now = datetime.now(timezone.utc)
            price = random.uniform(10, 500)
            size = random.uniform(1, 5)
            yield QuoteTick(
                ts=now,
                symbol=symbol,
                price=price,
                size=size,
                bid=price - 0.05,
                ask=price + 0.05,
                side="ASK" if random.random() > 0.5 else "BID",
                provider="simulated",
                kind=kind,
            )


async def simulated_quotes(symbols: Iterable[str]) -> AsyncIterator[QuoteTick]:
    async for tick in _simulated_stream(symbols, "quotes"):
        yield tick


async def simulated_trades(symbols: Iterable[str]) -> AsyncIterator[QuoteTick]:
    async for tick in _simulated_stream(symbols, "trades"):
        yield tick


try:  # pragma: no cover - optional imports
    from datafeed.crypto import CryptoFeed
    from datafeed.equities import EquityFeed
except Exception:  # pragma: no cover - degrade gracefully
    EquityFeed = None  # type: ignore
    CryptoFeed = None  # type: ignore


def equities_feed() -> BaseFeed | None:
    settings = load_settings()
    if EquityFeed is None:
        return None
    feed = EquityFeed(settings=settings)
    if feed.mode == "simulated":
        LOGGER.warning("Equities feed running in simulated mode")
    return feed


def crypto_feed() -> BaseFeed | None:
    settings = load_settings()
    if CryptoFeed is None:
        return None
    feed = CryptoFeed(settings=settings)
    if feed.mode == "simulated":
        LOGGER.warning("Crypto feed running in simulated mode")
    return feed

