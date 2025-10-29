from __future__ import annotations

from datetime import datetime, timedelta

import ccxt
import pandas as pd
import yfinance as yf

from .config import load_settings


def _resolve_start_date(days: int) -> str:
    start_dt = datetime.utcnow() - timedelta(days=days)
    return start_dt.strftime("%Y-%m-%d")


def _intraday_period(days: int) -> str:
    """Clamp intraday history to a supported window for yfinance."""

    if days <= 7:
        return "7d"
    if days <= 30:
        return "30d"
    if days <= 60:
        return "60d"
    return "60d"


def load_equity(symbol: str = "SPY", days: int = 365 * 3, interval: str = "1d") -> pd.DataFrame:
    intraday_intervals: set[str] = {"1h", "30m", "15m", "5m", "1m"}
    if interval in intraday_intervals:
        period = _intraday_period(days)
        df = yf.download(
            symbol,
            period=period,
            interval=interval,
            auto_adjust=False,
            progress=False,
        )
    else:
        start = _resolve_start_date(days)
        df = yf.download(
            symbol,
            start=start,
            interval=interval,
            auto_adjust=False,
            progress=False,
        )
    if df.empty:
        raise ValueError(f"No data for {symbol}.")
    df = df.rename(columns=str.lower).dropna()
    return df


def load_crypto(
    symbol: str = "BTC/USDT",
    exchange: str = "binance",
    timeframe: str = "1d",
    limit: int = 1000,
) -> pd.DataFrame:
    settings = load_settings()
    exchange_ctor = getattr(ccxt, exchange)
    client_args = {}
    if settings.BINANCE_API_KEY and settings.BINANCE_API_SECRET:
        client_args.update({
            "apiKey": settings.BINANCE_API_KEY,
            "secret": settings.BINANCE_API_SECRET,
        })
    ex = exchange_ctor(client_args)
    ohlcv = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    if not ohlcv:
        raise ValueError(f"No OHLCV for {symbol} on {exchange}.")
    df = pd.DataFrame(ohlcv, columns=["time", "open", "high", "low", "close", "volume"])
    df["time"] = pd.to_datetime(df["time"], unit="ms")
    df = df.set_index("time").astype(float)
    return df
