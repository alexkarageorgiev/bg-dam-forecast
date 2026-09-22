"""Every setting the package uses, in one place."""
from pathlib import Path

import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent.parent
DATA_DIR   = ROOT / "data"
MODEL_DIR  = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"
DATASET    = DATA_DIR / "dam_dataset.parquet"      # cached training frame (committed)
MODEL_FILE = MODEL_DIR / "lgbm.joblib"
META_FILE  = MODEL_DIR / "meta.json"

# ── Market & location ────────────────────────────────────────────────────────
TZ   = "Europe/Sofia"
ZONE = "BG"
LAT, LON = 42.70, 23.32                            # Sofia, a national weather proxy
HISTORY_START = pd.Timestamp("2017-01-01", tz=TZ)  # ENTSO-E has no BG price before 2017
HORIZON = 48                                       # hours forecast: D+1 00:00 -> D+2 23:00

# ── External signals ─────────────────────────────────────────────────────────
NEIGHBOURS  = {"gr_price": "GR", "ro_price": "RO"}  # coupled day-ahead markets
COMMODITIES = {"gas_ttf": "TTF=F",                  # Dutch TTF gas, EUR/MWh
               "coal_api2": "MTF=F",                # API2 Rotterdam coal, USD/t
               "brent": "BZ=F",                     # Brent crude, USD/bbl
               "carbon_proxy": "KRBN"}              # carbon ETF — a PROXY for EU ETS
WEATHER_VARS = ["temperature_2m", "wind_speed_10m", "shortwave_radiation", "cloud_cover"]

# ── Feature blocks (see notebooks/02_model.ipynb, "Features") ────────────────
CAL_RAW    = ["hour", "dayofweek", "month", "is_weekend", "is_holiday"]       # trees
CAL_CYC    = ["hour_sin", "hour_cos", "month_sin", "month_cos",
              "is_weekend", "is_holiday"]                                     # linear / LSTM
PRICE_LAGS = ["price_lag_48h", "price_lag_72h", "price_lag_168h"]
ROLLING    = ["price_roll_std_7d", "price_roll_mean_30d", "price_trend_7d_30d"]
RAMP       = ["load_ramp_1h", "net_load_ramp_1h", "net_load_ramp_3h"]
DRIVERS    = ["net_load_mw", "load_fc_mw", "solar_fc_mw",
              "wind_onshore_fc_mw", "wind_offshore_fc_mw"]
WEATHER    = ["temp_c", "wind_speed_ms", "radiation_wm2", "cloud_cover_pct"]
NEIGHBOUR  = ["ro_price_lag_48h", "ro_price_lag_168h", "gr_price_lag_48h"]
# Coal (API2, MTF=F) is still fetched but not used: Yahoo stopped publishing it in
# Dec 2025, so a live forecast could never have it. Dropping it also lowered the MAE.
FUEL       = ["gas_ttf_lag_48h", "gas_ttf_chg_1w"]

# Raw columns a live forecast needs from the day-ahead publications.
DRIVER_RAW = ["load_fc_mw", "solar_fc_mw", "wind_onshore_fc_mw"]

TARGET       = "dam_price_eur_mwh"
BASELINE_COL = "price_roll_mean_30d"   # the 30-day level a "residual" model predicts around

# ── Model ────────────────────────────────────────────────────────────────────
TARGET_MODE   = "direct"               # "direct" beat "residual" on the holdout (02_model.ipynb)
TEST_FRACTION = 0.15                   # chronological holdout: the most recent 15%
LGBM_PARAMS = dict(n_estimators=800, learning_rate=0.03, num_leaves=31,
                   min_child_samples=50, reg_lambda=1.0,
                   subsample=0.8, colsample_bytree=0.8,
                   random_state=42, verbose=-1)
