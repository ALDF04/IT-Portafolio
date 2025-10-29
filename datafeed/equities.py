"""Equity market data feeds with graceful fallbacks."""
from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass
from datetime import datetime, timezone

import yfinance as yf

from datafeed import QuoteTick, simulated_quotes, simulated_trades
from utils.config import Settings

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class EquityFeed:
    settings: Settings
    poll_interval: float = 2.0

    def __post_init__(self) -> None:
        self.mode = self._resolve_mode()
        LOGGER.info("Equities feed initialised in %s mode", self.mode)

    def _resolve_mode(self) -> str:
        if self.settings.POLYGON_API_KEY:
            return "polygon"
        if self.settings.IEX_TOKEN:
            return "iex"
        if self.settings.ALPACA_API_KEY and self.settings.ALPACA_API_SECRET:
            return "alpaca"
        if self.settings.IBKR_GATEWAY_HOST and self.settings.IBKR_CLIENT_ID is not None:
            return "ibkr"
        return "simulated"

    async def subscribe_quotes(self, symbols: Iterable[str]) -> AsyncIterator[QuoteTick]:
        if self.mode == "simulated":
            async for tick in simulated_quotes(symbols):
                yield tick
            return

        async for tick in self._polling_stream(symbols, kind="quotes"):
            yield tick

    async def subscribe_trades(self, symbols: Iterable[str]) -> AsyncIterator[QuoteTick]:
        if self.mode == "simulated":
            async for tick in simulated_trades(symbols):
                yield tick
            return

        async for tick in self._polling_stream(symbols, kind="trades"):
            yield tick

    async def _polling_stream(self, symbols: Iterable[str], kind: str) -> AsyncIterator[QuoteTick]:
        """Fallback REST polling using yfinance closes."""

        symbols_list = list(symbols)
        last_prices: dict[str, float] = {}
        while True:
            data = yf.download(
                symbols_list,
                period="1d",
                interval="1m",
                progress=False,
                auto_adjust=False,
            )
            if isinstance(data, dict):  # pragma: no cover - safety for yfinance variants
                close_frame = data.get("Close")
            else:
                close_frame = data["Close"] if "Close" in data else None
            now = datetime.now(timezone.utc)
            if close_frame is None:
                LOGGER.warning("yfinance returned empty data for %s", symbols_list)
            else:
                for symbol in symbols_list:
                    try:
                        price = float(close_frame[symbol].dropna().iloc[-1])
                    except Exception:
                        continue
                    prev = last_prices.get(symbol, price)
                    side = "ASK" if price >= prev else "BID" if price < prev else "UNK"
                    tick = QuoteTick(
                        ts=now,
                        symbol=symbol,
                        price=price,
                        size=1.0,
                        bid=min(price, prev),
                        ask=max(price, prev),
                        side=side,
                        provider=self.mode,
                        kind=kind,
                    )
                    last_prices[symbol] = price
                    yield tick
            await asyncio.sleep(self.poll_interval)

