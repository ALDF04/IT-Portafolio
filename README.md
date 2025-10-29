# Terminal Cuant — Andrey
- EMAs 9/21/50/200 + EMA 21W sobre diario
- RVOL (Z-Score) + CVD simple
- Dashboard en Streamlit

## Codespaces
1) Abre Codespace del repo.
2) Se instalan deps automáticamente.
3) Ejecuta:
   ```bash
   cp .env.sample .env
   streamlit run dashboards/app_streamlit.py
   ```

Local (Windows)

```
py -3.11 -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
cp .env.sample .env
streamlit run dashboards/app_streamlit.py
```

Local (macOS/Linux)

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.sample .env
streamlit run dashboards/app_streamlit.py
```

## Live Data & Indicadores Pro
- Toggles en la barra lateral para activar aVWAP anual, aVWAP trimestral y POC (Point of Control) sobre el gráfico.
- Indicadores adicionales: ancho de Bandas de Bollinger + ADX (Squeeze), RSI y bandera de régimen (placeholder "normal"/"shock").
- Habilita "Refresco en vivo (5s)" para rerender automático.

## Proveedores RT y configuración
- Configura `.env` con credenciales opcionales para equities (`POLYGON_API_KEY`, `IEX_TOKEN`, `ALPACA_API_KEY/SECRET`, `IBKR_*`).
- Para cripto habilita `CCXTPRO=true` y ajusta `EXCHANGE` (por defecto `binance`). Sin credenciales se usa modo seguro simulado/REST.
- `REDIS_URL` activa snapshots en Redis; `TIMESCALE_URL` habilita escritura a TimescaleDB vía SQLAlchemy.

## Mapa de Liquidez (intermarket/rotación)
- Expander dedicado con RiskScore agregado (DXY, VIX, TNX, TLT) y badges coloreados según el sesgo (≥0.3 verde, ≤–0.3 rojo).
- Ratios intermarket clave (QQQ/SPY, SOXX/SPY, HYG/TLT, XLF/XLK, XLE/SPY) para evaluar flujos de riesgo y rotación.
- Lectura rápida de sectores (XLK, SOXX, XLE, XLF, XLP, XLI, XLV) con cambios porcentuales diarios.

## Overlays avanzados (aVWAP/POC/Order Flow)
- Activa aVWAP de sesión y aVWAP de evento (ingresa timestamp ISO8601 UTC) desde el sidebar.
- Visualiza POC global y POC de sesión simultáneamente; el panel de métricas reporta la distancia al POC en ATR.
- El order flow tick-a-tick (CVD agresor y VRVP) se gestiona con `orderflow/` y puede persistirse vía Redis/Parquet/Timescale.

## Mapa Sectorial & Watchlists Pro
- Pestaña "Mapa Sectorial" con heatmap 1D coloreado (`dashboards/app_streamlit.py`).
- Pestaña "Watchlist Pro" calcula bias multi-EMA, RiskScore global, RVOL_Z, slope CVD, distancia al POC y checklist 5× autocompletado. Marca "A+" si ≥4/5 checks.

## Cache opcional con Redis
Configura `REDIS_URL` en `.env` (por ejemplo `redis://localhost:6379/0`). Si no está definido, el dashboard sigue funcionando sin caché.

## Backtest rápido
Ejecuta el backtest mínimo con vectorbt:

```
python backtests/vbt_long_ema21_rvol.py
```

## Analista IA (manual)
- Registra snapshots en `data/snapshots/` y utiliza `notebooks/01_build_dataset.ipynb` para generar features + labels en `data/experiments/train.parquet`.
- Entrena y calibra el modelo desde `notebooks/02_train_explain.ipynb`, lo que actualiza `models/latest/` y produce `reports/weekly_lessons.md`.
- El panel "Analista IA" en Streamlit permite evaluar la última vela (botón **Evaluar IA**), revisar probabilidad `p(win)`, top-3 SHAP, setups similares y niveles sugeridos.
- Usa la casilla "Log candidate" para guardar la señal manualmente en `data/trades/candidates.csv`.
- La capa IA es solo soporte de decisión: no dispara órdenes ni ejecución automática.

## Algoritmos institucionales educativos
- **Momentum-Factor Model (BlackRock)**: mezcla componentes trend/carry/value usando la estructura de EMAs 21/50/200, RVOL y (si existe) `open_interest`. El score `momentum_score` aparece en el expander "Algoritmos Institucionales".
- **Hybrid Liquidity Index (Citadel)**: estima estrés de liquidez a partir de spread relativo, delta de volumen y densidad. El resultado `liquidity_stress` se muestra junto con sus componentes `liquidity_spread/liquidity_delta/liquidity_density`.
- **Volatility Adjusted Position Sizer (Bridgewater)**: ajusta el tamaño sugerido con `position_factor`, combinando ATR normalizado y RVOL.
- **AI-Driven Correlation Matrix (AQR)**: calcula correlaciones rolling 60d contra BTC, SPX, DXY (ETF UUP) y NVDA para detectar confluencias multi-activo.
- Todos los algoritmos son informativos y manuales; no generan ejecución automática y pueden ampliarse con datos de proveedores RT reales.

## Arquitectura de datos (Redis + Parquet + TimescaleDB)
- `storage/cache.py` gestiona snapshots en Redis (`set_snapshot`, `set_df`). Si no hay Redis, se degrada a memoria local.
- `storage/sink.py` guarda streams en Parquet (`data/streams/<symbol>/<date>.parquet`) y, si defines `TIMESCALE_URL`, envía los ticks/featuros a TimescaleDB.
- `datafeed/` abstrae proveedores Polygon/IEX/Alpaca/IBKR (equities) y ccxt.pro/REST (cripto), normalizando ticks con agresión BID/ASK.
