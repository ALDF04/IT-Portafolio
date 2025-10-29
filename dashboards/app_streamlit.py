import json
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from datafeed import crypto_feed, equities_feed
from indicators.avwap import (
    add_avwap_annual,
    add_avwap_event,
    add_avwap_quarterly,
    add_avwap_session,
)
from indicators.features import add_all
from indicators.institutional import apply_institutional_algorithms
from indicators.liquidity_map import liquidity_snapshot, sign_state
from indicators.signals_pro import add_regime_flags, add_rsi, add_squeeze_adx
from indicators.volprofile import volume_profile_poc
from ml.explain import explain_sample
from ml.featureset import latest_feature_row
from ml.similarity import SetupSimilarity
from orderflow.session import is_cash_session
from signals.rules import signal_alignment_21w, signal_ema_reclaim_rvol
from utils.data_sources import load_crypto, load_equity

st.set_page_config(page_title="Andrey Quant", layout="wide")


def _resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    return (
        df.resample(rule)
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        .dropna()
    )


@st.cache_data(ttl=900)
def _supports_equity_4h(symbol: str) -> bool:
    try:
        df = load_equity(symbol, days=30, interval="1h")
    except Exception:
        return False
    return not df.empty


@st.cache_data(ttl=300)
def _cached_liquidity_snapshot() -> dict[str, float | None]:
    return liquidity_snapshot()


BENCHMARK_TICKERS = {
    "BTC": "BTC-USD",
    "SPX": "^GSPC",
    "DXY": "UUP",
    "NVDA": "NVDA",
}


@st.cache_data(ttl=3600)
def _benchmark_closes(days: int = 365) -> dict[str, pd.Series]:
    series_map: dict[str, pd.Series] = {}
    for label, ticker in BENCHMARK_TICKERS.items():
        try:
            ref = load_equity(ticker, days=days, interval="1d")
        except Exception:
            if label == "BTC":
                try:
                    ref = load_equity("BTC-USD", days=days, interval="1d")
                except Exception:
                    continue
            else:
                continue
        series = ref["close"].astype(float)
        series_map[label] = series
    return series_map


def _badge(label: str, value: float | None) -> str:
    if value is None:
        color = "#6c757d"
        text = "—"
    else:
        pct_value = value * 100
        if value >= 0.3:
            color = "#198754"
        elif value <= -0.3:
            color = "#dc3545"
        else:
            color = "#fd7e14"
        text = f"{pct_value:.2f}%"
    return (
        f"<span style='display:inline-block;padding:0.2rem 0.6rem;border-radius:999px;"
        f"background:{color};color:white;margin:0.1rem 0.2rem;font-size:0.85rem;'>"
        f"{label}: {text}</span>"
    )


def _arrow(value: float | None) -> str:
    state = sign_state(value if value is not None else 0.0)
    if value is None:
        return "—"
    mapping = {1: "▲", -1: "▼", 0: "→"}
    return mapping.get(state, "→")


def _badge_entry(snapshot: dict, key: str, label: str) -> str:
    value = snapshot.get(key)
    return _badge(f"{label} {_arrow(value)}", value)


def _load_model_bundle() -> dict | None:
    model_dir = Path("models/latest")
    required = ["model.pkl", "scaler.pkl", "calibrator.pkl", "features.json"]
    if not all((model_dir / name).exists() for name in required):
        return None
    try:
        model = joblib.load(model_dir / "model.pkl")
        scaler = joblib.load(model_dir / "scaler.pkl")
        calibrator = joblib.load(model_dir / "calibrator.pkl")
        with (model_dir / "features.json").open("r", encoding="utf-8") as fh:
            meta = json.load(fh)
        feature_columns = meta.get("feature_columns", [])
        return {
            "model": model,
            "scaler": scaler,
            "calibrator": calibrator,
            "feature_columns": feature_columns,
        }
    except Exception as exc:  # pragma: no cover - defensive
        st.warning(f"No fue posible cargar el modelo IA: {exc}")
        return None


def _log_candidate(row: dict) -> None:
    trades_dir = Path("data/trades")
    trades_dir.mkdir(parents=True, exist_ok=True)
    path = trades_dir / "candidates.csv"
    try:
        import pandas as pd

        df = pd.DataFrame([row])
        if path.exists():
            df_existing = pd.read_csv(path)
            df = pd.concat([df_existing, df], ignore_index=True)
        df.to_csv(path, index=False)
    except Exception as exc:  # pragma: no cover - defensive
        st.error(f"No fue posible registrar candidato: {exc}")


def _suggest_levels(latest_row, atr: float, fallback_close: float) -> dict:
    ema21 = latest_row.get("ema_21")
    avwap_y = latest_row.get("avwap_y")
    poc = latest_row.get("poc")
    close = latest_row.get("close", fallback_close)
    atr_value = atr if atr and atr > 0 else 0.0

    invalidation_basis = [value for value in [avwap_y, ema21] if value]
    if invalidation_basis:
        invalidation_ref = min(invalidation_basis)
    else:
        invalidation_ref = close
    invalidation = invalidation_ref - 0.5 * atr_value if atr_value else invalidation_ref

    tp1_candidates = [value for value in [poc, avwap_y, close + atr_value] if value]
    tp1 = max(tp1_candidates) if tp1_candidates else close + atr_value

    tp2 = close + max(1.5 * atr_value, atr_value) if atr_value else close * 1.01

    return {
        "invalidation": invalidation,
        "tp1": tp1,
        "tp2": tp2,
    }


def _provider_status(asset_type: str) -> dict[str, str | float]:
    feed = equities_feed() if asset_type == "Equity/ETF" else crypto_feed()
    if feed is None:
        return {"provider": "offline", "mode": "simulado", "latency": "N/A"}
    mode = getattr(feed, "mode", "desconocido")
    provider = "ccxt.pro" if asset_type != "Equity/ETF" and mode != "rest" else mode
    return {"provider": provider, "mode": mode, "latency": "N/A"}


def _session_poc(df: pd.DataFrame) -> float | None:
    if not isinstance(df.index, pd.DatetimeIndex) or df.empty:
        return None
    last_day = df.index[-1].date()
    session_df = df[df.index.date == last_day]
    if session_df.empty:
        return None
    return volume_profile_poc(session_df)


def _atr_series(df: pd.DataFrame, length: int = 14) -> pd.Series:
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)
    components = [high - low, (high - prev_close).abs(), (low - prev_close).abs()]
    tr = pd.concat(components, axis=1).max(axis=1)
    return tr.rolling(length).mean()


def _parse_event_ts(raw: str) -> datetime | None:
    raw = raw.strip()
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return None


def _compute_watchlist(
    symbols: list[str], asset_type: str, rvol_min: float, snapshot: dict
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    risk_score = snapshot.get("risk_score")
    for sym in symbols:
        sym = sym.strip()
        if not sym:
            continue
        try:
            if asset_type == "Equity/ETF":
                data = load_equity(sym, days=365, interval="1d")
            else:
                data = load_crypto(sym, timeframe="1d", limit=365)
        except Exception as exc:
            rows.append({"symbol": sym.upper(), "error": str(exc)})
            continue
        data = add_all(data)
        data = add_squeeze_adx(data)
        data = add_rsi(data)
        data = add_regime_flags(data)
        data = add_avwap_annual(data)
        data = add_avwap_quarterly(data)
        data = add_avwap_session(data)
        data, _ = apply_institutional_algorithms(data)
        atr = _atr_series(data).iloc[-1]
        last = data.iloc[-1]
        poc_value = volume_profile_poc(data.tail(200))
        dist_to_poc = (last["close"] - poc_value) / atr if atr else np.nan
        cvd_slope = data["cvd"].diff().rolling(5).mean().iloc[-1]
        ema_stack = bool(
            last.get("ema_9", 0) > last.get("ema_21", 0)
            and last.get("close", 0) > last.get("ema_21", 0)
        )
        institutional_zone = bool(
            abs(last.get("close", 0) - (last.get("avwap_y", last["close"]) or last["close"]))
            < (atr or 1)
            or (poc_value is not None and abs(last.get("close", 0) - poc_value) < (atr or 1))
        )
        mtf_align = bool(
            last.get("ema_21", 0) > last.get("ema_50", 0) > last.get("ema_200", 0)
            if last.get("ema_200")
            else False
        )
        checklist = {
            "ema_stack": ema_stack,
            "rvol": bool(last.get("rvol_z", 0) > rvol_min),
            "cvd": bool(cvd_slope > 0),
            "institutional": institutional_zone,
            "mtf": mtf_align,
        }
        score = int(sum(checklist.values()))
        rows.append(
            {
                "symbol": sym.upper(),
                "bias": "Bull" if checklist["ema_stack"] else "Neutral/Bear",
                "riskScore": risk_score,
                "rvol_z": float(last.get("rvol_z", 0.0)),
                "cvd_slope": float(cvd_slope),
                "dist_to_poc": float(dist_to_poc) if dist_to_poc == dist_to_poc else np.nan,
                "momentum_score": float(last.get("momentum_score", 0.0)),
                "liquidity_stress": float(last.get("liquidity_stress", 0.0)),
                "position_factor": float(last.get("position_factor", 0.0)),
                "checklist_5x": f"{score}/5",
                "grade": "A+" if score >= 4 else "B" if score == 3 else "Observa",
                "poc": poc_value,
                "ema_21": last.get("ema_21"),
                "ema_21w": last.get("ema_21w"),
                "avwap_y": last.get("avwap_y"),
                "checklist": checklist,
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "symbol",
                "bias",
                "riskScore",
                "rvol_z",
                "cvd_slope",
                "dist_to_poc",
                "checklist_5x",
                "grade",
            ]
        )
    df_watch = pd.DataFrame(rows)
    return df_watch


def _sector_heatmap(snapshot: dict[str, float | None]) -> pd.DataFrame:
    sectors = ["xlk", "soxx", "xle", "xlf", "xlp", "xli", "xlv"]
    data = {
        "Ticker": [sector.upper() for sector in sectors],
        "% 1D": [snapshot.get(f"{sector}_pct") for sector in sectors],
    }
    return pd.DataFrame(data)

supports_equity_4h = False

with st.sidebar:
    st.title("Terminal Cuant — Andrey")
    asset_type = st.selectbox("Tipo", ["Equity/ETF", "Crypto"])
    symbol = st.text_input("Símbolo", "SPY" if asset_type == "Equity/ETF" else "BTC/USDT")
    if asset_type == "Equity/ETF":
        supports_equity_4h = _supports_equity_4h(symbol)
        timeframe_options = ["1d"]
        if supports_equity_4h:
            timeframe_options.append("4h")
        timeframe_options.append("1w")
    else:
        timeframe_options = ["1d", "4h", "1h"]
    default_index = timeframe_options.index("1d") if "1d" in timeframe_options else 0
    tf = st.selectbox("Timeframe", timeframe_options, index=default_index)
    days = st.slider("Días a cargar", 200, 1500, 500)
    rvol_min = st.slider("Umbral RVOL (Z)", 0.5, 3.0, 1.3, 0.1)
    show_avwap_y = st.checkbox("Mostrar aVWAP anual", value=False)
    show_avwap_q = st.checkbox("Mostrar aVWAP trimestral", value=False)
    show_avwap_session = st.checkbox("Mostrar aVWAP sesión", value=False)
    event_raw = st.text_input("Evento aVWAP (ISO8601 UTC)", value="")
    show_avwap_event = st.checkbox("Mostrar aVWAP evento", value=False)
    show_poc = st.checkbox("Mostrar POC", value=False)
    show_session_poc = st.checkbox("Mostrar POC sesión", value=False)
    watchlist_input = st.text_input("Watchlist (coma)", value="SPY, QQQ, BTC/USDT")
    live = st.checkbox("Refresco en vivo (5s)", value=False)

if asset_type == "Equity/ETF":
    tf_display = tf
    if tf == "1d":
        df = load_equity(symbol, days=days, interval="1d")
    elif tf == "4h":
        try:
            df_1h = load_equity(symbol, days=min(days, 60), interval="1h")
        except Exception as exc:
            st.warning(f"No fue posible obtener 4H, usando 1D. Detalle: {exc}")
            df = load_equity(symbol, days=days, interval="1d")
            tf_display = "1d"
        else:
            df = _resample_ohlc(df_1h, "4H")
            tf_display = "4h"
    else:  # 1w
        daily = load_equity(symbol, days=days, interval="1d")
        df = _resample_ohlc(daily, "1W")
        tf_display = "1w"
else:
    df = load_crypto(symbol, timeframe=tf, limit=min(days, 1500))
    tf_display = tf

df["symbol"] = symbol
df["timeframe"] = tf_display

df = add_all(df)
df = add_squeeze_adx(df)
df = add_rsi(df)
df = add_regime_flags(df)
df = add_avwap_annual(df)
df = add_avwap_quarterly(df)
df = add_avwap_session(df)
event_ts = _parse_event_ts(event_raw)
if show_avwap_event and event_ts:
    df = add_avwap_event(df, event_ts)

poc_value = volume_profile_poc(df)
df["poc"] = poc_value
session_poc = _session_poc(df)
df["atr"] = _atr_series(df)
snapshot = _cached_liquidity_snapshot()
benchmarks = _benchmark_closes(days=min(days, 730))
df, correlations = apply_institutional_algorithms(df, benchmarks)
provider_info = _provider_status(asset_type)
session_mode = "Cash" if is_cash_session(df.index[-1]) else "Globex"
status_text = (
    f"Proveedor activo: {provider_info['provider']} "
    f"(modo {provider_info['mode']}) — Sesión: {session_mode}"
)
st.caption(status_text)

st.subheader(f"{symbol} — {tf_display}")
fig = go.Figure()
fig.add_trace(
    go.Candlestick(
        x=df.index,
        open=df.open,
        high=df.high,
        low=df.low,
        close=df.close,
        name="Precio",
    )
)
for col in ["ema_9", "ema_21", "ema_50", "ema_200"]:
    if col in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df[col], name=col, line=dict(width=1)))
if "ema_21w" in df.columns:
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["ema_21w"],
            name="ema_21w",
            line=dict(width=1, dash="dot"),
        )
    )
if show_avwap_y and "avwap_y" in df.columns:
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["avwap_y"],
            name="aVWAP anual",
            line=dict(width=1, dash="dot"),
        )
    )
if show_avwap_q:
    for qm in [1, 4, 7, 10]:
        col = f"avwap_q{qm}"
        if col in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df[col],
                    name=f"aVWAP Q{qm}",
                    line=dict(width=1, dash="dot"),
                )
            )
if show_avwap_session and "avwap_session" in df.columns:
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["avwap_session"],
            name="aVWAP sesión",
            line=dict(width=1, dash="dashdot"),
        )
    )
if show_avwap_event and event_ts and "avwap_event" in df.columns:
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["avwap_event"],
            name="aVWAP evento",
            line=dict(width=1, dash="longdash"),
        )
    )
if show_poc and poc_value is not None:
    fig.add_hline(y=poc_value, line_dash="dot", line_color="orange", annotation_text="POC")
if show_session_poc and session_poc is not None:
    fig.add_hline(
        y=session_poc,
        line_dash="dash",
        line_color="purple",
        annotation_text="POC sesión",
    )

st.plotly_chart(fig, use_container_width=True)

last_row = df.iloc[-1]
atr_latest = last_row.get("atr") or _atr_series(df).iloc[-1]
dist_session_atr = (
    (last_row.get("close", 0) - session_poc) / atr_latest if session_poc and atr_latest else None
)
metrics_cols = st.columns(3)
dist_display = f"{dist_session_atr:.2f}" if dist_session_atr else "—"
metrics_cols[0].metric("Distancia POC (ATR)", dist_display)
metrics_cols[1].metric(
    "RVOL Z",
    f"{last_row.get('rvol_z', float('nan')):.2f}" if last_row.get("rvol_z") is not None else "—",
)
metrics_cols[2].metric("RiskScore", f"{snapshot.get('risk_score', float('nan')):.2f}")

sector_tab, watch_tab = st.tabs(["Mapa Sectorial", "Watchlist Pro"])
with sector_tab:
    st.caption("Rotación sectorial 1D")
    heatmap_df = _sector_heatmap(snapshot)
    if not heatmap_df.empty:
        st.dataframe(heatmap_df.style.background_gradient(subset=["% 1D"], cmap="RdYlGn"))
    else:
        st.info("Sin datos de sectores disponibles")

with watch_tab:
    st.caption("Checklist 5× autocompletado (manual)")
    symbols = [s.strip() for s in watchlist_input.split(",") if s.strip()]
    watch_df = _compute_watchlist(symbols, asset_type, rvol_min, snapshot)
    if watch_df.empty:
        st.info("Agrega símbolos en el sidebar para evaluar la watchlist")
    else:
        checklist_details = None
        if "checklist" in watch_df.columns:
            checklist_details = watch_df.pop("checklist")
        st.dataframe(watch_df)
        if checklist_details is not None:
            st.json(checklist_details.to_dict(), expanded=False)

c1, c2 = st.columns(2)
with c1:
    st.subheader("RVOL (Z-Score)")
    st.line_chart(df["rvol_z"])
with c2:
    st.subheader("CVD (simple)")
    st.line_chart(df["cvd"])

regime = df["regime"].iloc[-1] if "regime" in df.columns else "Desconocido"
st.markdown("### Régimen")
st.info(f"Régimen actual: {regime}")

with st.expander("Algoritmos Institucionales", expanded=False):
    st.caption(
        "Réplicas educativas de modelos Momentum (BlackRock), Liquidez (Citadel), "
        "Sizer volátil (Bridgewater) y correlación IA (AQR)."
    )
    momentum = last_row.get("momentum_score")
    liquidity = last_row.get("liquidity_stress")
    position_factor = last_row.get("position_factor")
    col_mom, col_liq, col_pos = st.columns(3)
    col_mom.metric("Momentum Score", f"{momentum:.2f}" if pd.notna(momentum) else "—")
    col_liq.metric("Liquidity Stress", f"{liquidity:.2f}" if pd.notna(liquidity) else "—")
    col_pos.metric(
        "Position Factor",
        f"{position_factor:.2f}" if pd.notna(position_factor) else "—",
    )

    components = {
        "Trend": last_row.get("momentum_trend"),
        "Carry": last_row.get("momentum_carry"),
        "Value": last_row.get("momentum_value"),
        "Spread": last_row.get("liquidity_spread"),
        "Delta": last_row.get("liquidity_delta"),
        "Density": last_row.get("liquidity_density"),
    }
    st.markdown("#### Componentes clave")
    st.write(pd.Series(components).to_frame(name="Valor"))

    if correlations:
        st.markdown("#### Correlación 60d (AI Confluence)")
        corr_df = (
            pd.DataFrame.from_dict(correlations, orient="index", columns=["corr"])
            .sort_index()
        )
        st.table(corr_df.style.format({"corr": "{:.2f}"}))
    else:
        st.info("Sin datos de correlación disponibles")

with st.expander("Mapa de Liquidez", expanded=False):
    snapshot_local = snapshot
    st.caption("Referencias de rotación y liquidez intermarket")
    risk_items = [
        _badge_entry(snapshot_local, "risk_score", "RiskScore"),
        _badge_entry(snapshot_local, "dxy_pct", "DXY"),
        _badge_entry(snapshot_local, "vix_pct", "VIX"),
        _badge_entry(snapshot_local, "tnx_pct", "TNX"),
        _badge_entry(snapshot_local, "tlt_pct", "TLT"),
    ]
    st.markdown("#### Riesgo macro")
    st.markdown(" ".join(risk_items), unsafe_allow_html=True)

    ratio_items = [
        _badge_entry(snapshot_local, "qqq_spy_pct", "QQQ/SPY"),
        _badge_entry(snapshot_local, "soxx_spy_pct", "SOXX/SPY"),
        _badge_entry(snapshot_local, "hyg_tlt_pct", "HYG/TLT"),
        _badge_entry(snapshot_local, "xlf_xlk_pct", "XLF/XLK"),
        _badge_entry(snapshot_local, "xle_spy_pct", "XLE/SPY"),
    ]
    st.markdown("#### Ratios intermarket")
    st.markdown(" ".join(ratio_items), unsafe_allow_html=True)

    sector_items = [
        _badge_entry(snapshot_local, "xlk_pct", "XLK"),
        _badge_entry(snapshot_local, "soxx_pct", "SOXX"),
        _badge_entry(snapshot_local, "xle_pct", "XLE"),
        _badge_entry(snapshot_local, "xlf_pct", "XLF"),
        _badge_entry(snapshot_local, "xlp_pct", "XLP"),
        _badge_entry(snapshot_local, "xli_pct", "XLI"),
        _badge_entry(snapshot_local, "xlv_pct", "XLV"),
    ]
    st.markdown("#### Sectoriales")
    st.markdown(" ".join(sector_items), unsafe_allow_html=True)

st.markdown("### Señal rápida")
try:
    dfx = signal_alignment_21w(
        signal_ema_reclaim_rvol(df.copy(), rvol_min=rvol_min),
        rvol_min=rvol_min,
    )
    last = dfx.tail(1).iloc[0]
    st.write(
        {
            "long_signal": int(last.get("long_signal", 0)),
            "aplus_signal": int(last.get("aplus_signal", 0)),
            "close": float(last["close"]),
            "ema_21": float(last["ema_21"]),
            "ema_21w": (
                float(last.get("ema_21w", last["ema_21"]))
                if "ema_21w" in last
                else float(last["ema_21"])
            ),
            "rvol_z": float(last["rvol_z"]),
        }
    )
except Exception as e:
    st.warning(f"No fue posible calcular señal rápida: {e}")

with st.expander("Analista IA", expanded=False):
    st.caption("Evaluación asistida por IA (sin ejecución automática)")
    log_candidate = st.checkbox("Log candidate", value=False, key="log_candidate")
    evaluate = st.button("Evaluar IA", type="primary")

    if evaluate:
        bundle = _load_model_bundle()
        if bundle is None:
            st.warning("Modelo IA no disponible. Entrena desde notebooks/02_train_explain.ipynb")
        else:
            try:
                feature_row = latest_feature_row(df)
            except Exception as exc:
                st.error(f"No fue posible generar features: {exc}")
            else:
                expected_cols = bundle["feature_columns"]
                row_df = feature_row.to_frame().T
                if expected_cols:
                    row_df = row_df.reindex(columns=expected_cols, fill_value=0.0)
                try:
                    proba = float(bundle["calibrator"].predict_proba(row_df)[:, 1])
                except Exception as exc:  # pragma: no cover - runtime safety
                    st.error(f"No fue posible inferir probabilidad: {exc}")
                    proba = None

                if proba is not None:
                    st.metric("p(win)", f"{proba:.2%}")
                    st.progress(min(max(proba, 0.0), 1.0))

                    shap_info = explain_sample(feature_row)
                    if "values" in shap_info:
                        shap_series = (
                            pd.Series(shap_info["values"]).abs().sort_values(ascending=False).head(3)
                        )
                        st.subheader("Top-3 SHAP")
                        st.write(shap_series)
                    else:
                        st.info(shap_info.get("message", "SHAP no disponible"))

                    similar_cases = []
                    data_path = Path("data/experiments/train.parquet")
                    if data_path.exists():
                        import pandas as pd

                        try:
                            dataset = pd.read_parquet(data_path)
                            feature_cols = bundle["feature_columns"] or list(row_df.columns)
                            dataset_features = dataset.reindex(columns=feature_cols).dropna()
                            if not dataset_features.empty:
                                similarity = SetupSimilarity(dataset_features)
                                matches = similarity.find_similar(feature_row, top_k=3)
                                for idx, score in matches:
                                    meta_row = dataset.loc[dataset_features.index[idx]]
                                    similar_cases.append(
                                        {
                                            "timestamp": str(
                                                meta_row.get("timestamp", meta_row.name)
                                            ),
                                            "symbol": meta_row.get("symbol", symbol),
                                            "outcome": meta_row.get("y_outcome", "?"),
                                            "similarity": score,
                                        }
                                    )
                        except Exception as exc:  # pragma: no cover
                            st.warning(f"No fue posible calcular similitud: {exc}")
                    if similar_cases:
                        st.subheader("Setups similares")
                        st.table(similar_cases)
                    else:
                        st.info("Sin setups similares disponibles")

                    st.markdown("#### Liquidez (features)")
                    liquidity_badges = [
                        _badge("RiskScore", feature_row.get("risk_score")),
                        _badge("QQQ/SPY", feature_row.get("qqq_spy_pct")),
                        _badge("SOXX/SPY", feature_row.get("soxx_spy_pct")),
                        _badge("HYG/TLT", feature_row.get("hyg_tlt_pct")),
                        _badge("XLF/XLK", feature_row.get("xlf_xlk_pct")),
                        _badge("XLE/SPY", feature_row.get("xle_spy_pct")),
                    ]
                    st.markdown(" ".join(liquidity_badges), unsafe_allow_html=True)

                    latest_row = df.iloc[-1]
                    levels = _suggest_levels(
                        latest_row,
                        feature_row.get("atr", latest_row.get("atr", 0.0)),
                        latest_row["close"],
                    )
                    st.subheader("Niveles sugeridos")
                    st.write(levels)

                    if log_candidate:
                        candidate = {
                            "ts": latest_row.name.isoformat(),
                            "symbol": symbol,
                            "tf": tf_display,
                            "features": json.dumps(feature_row.to_dict()),
                            "p_win": proba,
                            "invalidation": levels["invalidation"],
                            "tp1": levels["tp1"],
                            "tp2": levels["tp2"],
                        }
                        _log_candidate(candidate)
                        st.success("Candidato registrado en data/trades/candidates.csv")

if live:
    st.session_state["live_refresh"] = st.session_state.get("live_refresh", 0) + 1
    time.sleep(5)
    st.experimental_rerun()
