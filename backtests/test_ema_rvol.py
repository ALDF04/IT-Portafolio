from indicators.features import add_all
from signals.rules import signal_ema_reclaim_rvol
from utils.data_sources import load_equity


def main() -> None:
    try:
        df = load_equity("SPY", days=365 * 3, interval="1d")
        df = signal_ema_reclaim_rvol(add_all(df))
        print(df.tail())
    except Exception as exc:  # pragma: no cover - network dependency
        print(f"No fue posible ejecutar backtest mínimo: {exc}")


if __name__ == "__main__":
    main()
