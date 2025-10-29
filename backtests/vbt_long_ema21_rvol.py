import vectorbt as vbt

from indicators.features import add_all
from signals.rules import signal_ema_reclaim_rvol
from utils.data_sources import load_equity


def run_backtest(symbol: str = "SPY", years: int = 3) -> dict:
    df = load_equity(symbol, days=365 * years, interval="1d")
    df = add_all(df)
    df = signal_ema_reclaim_rvol(df)
    entries = df["long_signal"].astype(bool)
    exits = ~entries
    close = df["close"]

    if close.empty:
        return {"CAGR": None, "MaxDD": None, "WinRate": None, "ProfitFactor": None}

    portfolio = vbt.Portfolio.from_signals(close, entries, exits)
    stats = portfolio.stats()
    return {
        "CAGR": float(stats.get("CAGR", float("nan"))),
        "MaxDD": float(stats.get("Max Drawdown", float("nan"))),
        "WinRate": float(stats.get("Win Rate", float("nan"))),
        "ProfitFactor": float(stats.get("Profit Factor", float("nan"))),
    }


if __name__ == "__main__":
    results = run_backtest()
    print("Resultados backtest (placeholders si faltan datos):")
    for key, value in results.items():
        print(f"{key}: {value}")
