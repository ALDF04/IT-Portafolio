"""Crypto data feed abstraction built on top of ccxt/ccxt.pro."""
from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass
from datetime import datetime, timezone

import ccxt

try:  # pragma: no cover - optional dependency
    import ccxt.pro as ccxtpro
except Exception:  # pragma: no cover - gracefully degrade
    ccxtpro = None

from datafeed import QuoteTick, simulated_quotes, simulated_trades
from utils.config import Settings

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class CryptoFeed:
    settings: Settings
    poll_interval: float = 1.0

    def __post_init__(self) -> None:
        self.exchange_id = (self.settings.EXCHANGE or "binance").lower()
        self.mode = self._resolve_mode()
        self.exchange = getattr(ccxt, self.exchange_id)()
        LOGGER.info("Crypto feed initialised in %s mode for %s", self.mode, self.exchange_id)
        self._ws_client = None
        if self.mode == "ccxtpro" and ccxtpro is not None:
            try:
                self._ws_client = getattr(ccxtpro, self.exchange_id)()
            except Exception as exc:  # pragma: no cover - fallback safety
                LOGGER.warning("ccxt.pro client failed (%s), falling back to REST", exc)
                self.mode = "rest"

    def _resolve_mode(self) -> str:
        if self.settings.CCXTPRO and ccxtpro is not None:
            return "ccxtpro"
        return "rest"

    async def subscribe_quotes(self, symbols: Iterable[str]) -> AsyncIterator[QuoteTick]:
        if self.mode == "ccxtpro" and self._ws_client is not None:
            async for tick in self._ws_quotes(symbols):
                yield tick
            return

        if self.mode == "rest":
            async for tick in self._rest_poll(symbols, kind="quotes"):
                yield tick
            return

        async for tick in simulated_quotes(symbols):
            yield tick

    async def subscribe_trades(self, symbols: Iterable[str]) -> AsyncIterator[QuoteTick]:
        if self.mode == "ccxtpro" and self._ws_client is not None:
            async for tick in self._ws_trades(symbols):
                yield tick
            return

        if self.mode == "rest":
            async for tick in self._rest_poll(symbols, kind="trades"):
                yield tick
            return

        async for tick in simulated_trades(symbols):
            yield tick

    async def _ws_quotes(
        self, symbols: Iterable[str]
    ) -> AsyncIterator[QuoteTick]:  # pragma: no cover
        assert self._ws_client is not None
        for symbol in symbols:
            await self._ws_client.watch_order_book(symbol)
        while True:
            for symbol in symbols:
                order_book = await self._ws_client.watch_order_book(symbol)
                now = datetime.now(timezone.utc)
                bid = order_book["bids"][0][0] if order_book["bids"] else None
                ask = order_book["asks"][0][0] if order_book["asks"] else None
                price = ask if ask is not None else bid
                if price is None:
                    continue
                yield QuoteTick(
                    ts=now,
                    symbol=symbol,
                    price=float(price),
                    size=1.0,
                    bid=float(bid) if bid else None,
                    ask=float(ask) if ask else None,
                    side="ASK",
                    provider="ccxtpro",
                    kind="quotes",
                )

    async def _ws_trades(
        self, symbols: Iterable[str]
    ) -> AsyncIterator[QuoteTick]:  # pragma: no cover
        assert self._ws_client is not None
        for symbol in symbols:
            await self._ws_client.watch_trades(symbol)
        while True:
            for symbol in symbols:
                trades = await self._ws_client.watch_trades(symbol)
                for trade in trades:
                    is_buy = trade.get("side") == "buy"
                    side = "ASK" if trade.get("takerOrMaker") == "taker" and is_buy else "BID"
                    yield QuoteTick(
                        ts=datetime.fromtimestamp(trade["timestamp"] / 1000, tz=timezone.utc),
                        symbol=symbol,
                        price=float(trade["price"]),
                        size=float(trade.get("amount", trade.get("quantity", 0.0))),
                        bid=None,
                        ask=None,
                        side=side,
                        provider="ccxtpro",
                        kind="trades",
                    )

    async def _rest_poll(self, symbols: Iterable[str], kind: str) -> AsyncIterator[QuoteTick]:
        symbols_list = list(symbols)
        while True:
            for symbol in symbols_list:
                try:
                    ticker = self.exchange.fetch_ticker(symbol)
                except Exception as exc:  # pragma: no cover - defensive
                    LOGGER.warning("REST ticker failed for %s: %s", symbol, exc)
                    continue
                timestamp = ticker.get("timestamp", self.exchange.milliseconds())
                now = datetime.fromtimestamp(timestamp / 1000, tz=timezone.utc)
                price = float(ticker.get("last") or ticker.get("close") or 0.0)
                bid = ticker.get("bid")
                ask = ticker.get("ask")
                side = "UNK"
                if bid is not None and ask is not None and price:
                    if price >= float(ask):
                        side = "ASK"
                    elif price <= float(bid):
                        side = "BID"
                tick = QuoteTick(
                    ts=now,
                    symbol=symbol,
                    price=price,
                    size=float(ticker.get("baseVolume") or ticker.get("quoteVolume") or 0.0),
                    bid=float(bid) if bid else None,
                    ask=float(ask) if ask else None,
                    side=side,
                    provider=self.mode,
                    kind=kind,
                )
                yield tick
            await asyncio.sleep(self.poll_interval)

