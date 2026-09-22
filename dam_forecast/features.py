"""Feature engineering — the leakage rule lives here.

Lifted unchanged from the research notebook (notebooks/02_model.ipynb, "Features"),
so the model evaluated there is exactly the model forecast.py runs.

THE LEAKAGE RULE: the forecast is made at 08:00 for up to 48 h ahead, so a feature
may only use what is known at that moment.
  * Day-ahead forecasts (load, wind, solar) and weather -> used at their own timestamp.
  * Anything price-derived (BG, GR, RO prices, fuel closes) -> lags of >= 48 h only.
"""
import holidays as holidays_lib
import numpy as np
import pandas as pd

from . import config

# entsoe-py and Open-Meteo return their own column names; map them to snake_case.
RENAME_MAP = {
    "Forecasted Load":     "load_fc_mw",
    "Solar":               "solar_fc_mw",
    "Wind Onshore":        "wind_onshore_fc_mw",
    "Wind Offshore":       "wind_offshore_fc_mw",
    "temperature_2m":      "temp_c",
    "wind_speed_10m":      "wind_speed_ms",
    "shortwave_radiation": "radiation_wm2",
    "cloud_cover":         "cloud_cover_pct",
}


def standardize(df):
    """Raw merged frame -> one uniform hourly grid with tidy column names + net load.

    BG moved to 15-minute prices on 2025-10-01. Row-based lags only mean "hours" on
    an hourly grid, so the quarter-hours are averaged into hourly means first.
    """
    df = df.sort_index().resample("1h").mean()
    df = df.rename(columns={k: v for k, v in RENAME_MAP.items() if k in df.columns})
    # Net load = demand left for dispatchable (mostly fossil) plants after wind and
    # solar. Price tracks this residual far more closely than raw demand.
    renew = [c for c in ["solar_fc_mw", "wind_onshore_fc_mw", "wind_offshore_fc_mw"]
             if c in df.columns]
    if "load_fc_mw" in df.columns and renew:
        df["net_load_mw"] = df["load_fc_mw"] - df[renew].sum(axis=1)
    return df


def make_features(df):
    """Turn the raw hourly table (price target + day-ahead inputs) into a
    model-ready feature matrix — a SUPERSET of candidate columns. Which columns
    each model actually uses is decided next cell (trees vs. linear/LSTM want
    different encodings), but every column here obeys one hard rule:

    THE LEAKAGE RULE — a feature may only use information known at 08:00, the
    moment the 48 h-ahead forecast is made.
      • Day-ahead FORECASTS (load, wind, solar) and weather forecasts are
        published before the auction, so they are fair game at their own
        timestamp — that is the whole reason we collected forecasts, not actuals.
      • Past PRICES arrive with a delay. For a 48 h horizon, the newest price we
        can safely rely on for EVERY target hour in the window is 48 h old — so
        price-derived features start at 48 h and never at 24 h.
    One pure function means training and live inference share code and can never
    silently diverge.
    """
    out = df.copy()
    idx = out.index

    # ── Calendar features (always known in advance → free and safe) ─────────
    # Two encodings on purpose: the RAW integers below are for the TREES (they
    # split on them directly); the sin/cos pairs further down are for the LINEAR
    # / LSTM models, which need smooth cyclical inputs rather than magnitudes.
    out["hour"]       = idx.hour
    out["dayofweek"]  = idx.dayofweek                 # 0 = Monday
    out["month"]      = idx.month
    out["is_weekend"] = (idx.dayofweek >= 5).astype(int)

    # Bulgarian public holidays depress industrial demand and behave like
    # Sundays, whatever weekday they land on.
    bg_holidays = holidays_lib.Bulgaria(years=range(idx.year.min(), idx.year.max() + 1))
    out["is_holiday"] = idx.to_series().dt.date.apply(lambda d: d in bg_holidays).astype(int).values

    # Cyclical encoding: as raw integers, hour 23 and hour 0 look far apart even
    # though they are adjacent in time. sin/cos pairs restore that adjacency —
    # essential for the linear baseline and the LSTM (trees ignore them, so we
    # feed the raw integers to the trees and the sin/cos to the linear models).
    out["hour_sin"]  = np.sin(2 * np.pi * out["hour"]  / 24)
    out["hour_cos"]  = np.cos(2 * np.pi * out["hour"]  / 24)
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)

    # ── Lagged price features (leakage-safe: every lag ≥ 48 h) ──────────────
    # "What was the price at this hour 2 days / 3 days / a week ago?" The 168 h
    # (7-day) lag carries the weekly shape; 48 h / 72 h carry the most recent
    # level we are allowed to see.
    price = out["dam_price_eur_mwh"]
    out["price_lag_48h"]  = price.shift(48)
    out["price_lag_72h"]  = price.shift(72)
    out["price_lag_168h"] = price.shift(168)

    # Rolling statistics over windows ENDING 48 h before the target hour. The
    # .shift(48) BEFORE .rolling() keeps the window from peeking at anything
    # newer than 48 h — the same leakage rule as the lags.
    past = price.shift(48)
    out["price_roll_mean_7d"]  = past.rolling(168).mean()   # kept only to build the trend below
    out["price_roll_std_7d"]   = past.rolling(168).std()    # recent VOLATILITY (spike context)
    # A 30-day mean is a far steadier "regime level" than a single week — a hot
    # week in July says little about the price regime, a month says a lot. The
    # 7d-minus-30d spread is an explicit up/down TREND signal, and because it is
    # a DIFFERENCE it stays in-range for trees even as the market drifts (trees
    # can't extrapolate a rising level, but they can read "trend is positive").
    # price_roll_mean_30d also doubles as the BASELINE the models detrend against.
    out["price_roll_mean_30d"] = past.rolling(720).mean()   # ~30d level / regime
    out["price_trend_7d_30d"]  = out["price_roll_mean_7d"] - out["price_roll_mean_30d"]

    # ── Demand-ramp features (leakage-safe: day-ahead LOAD forecast is known) ─
    # The evening peak is where the model loses precision: a tree can't print a
    # value above what the lags suggest, so steep climbs get clipped. But the
    # day-ahead load / net-load forecast already encodes the ramp SHAPE and is
    # known at 08:00, so its hour-over-hour gradient is a leakage-safe pointer to
    # exactly those climbs — using the forecast at its own timestamp, no future.
    if "load_fc_mw" in out.columns:
        out["load_ramp_1h"] = out["load_fc_mw"].diff(1)
    if "net_load_mw" in out.columns:
        out["net_load_ramp_1h"] = out["net_load_mw"].diff(1)
        out["net_load_ramp_3h"] = out["net_load_mw"].diff(3)

    # ── External market signals (leakage-safe: all shifted ≥ 48 h) ──────────
    # Only added if the external-data cell ran, so make_features stays runnable
    # on its own. Romania couples tightly with BG (its lags carry real gain), so
    # we keep RO at 48 h AND 168 h but GR only at 48 h — GR's week-old lag scored
    # near-zero by gain. Brent and the carbon ETF are intentionally NOT lagged
    # into features: both scored low gain / high split (noise-fitting) and the
    # carbon proxy is missing pre-2020. Gas and coal (BG's real marginal fuels)
    # get a weekly-MOMENTUM term so the model sees them TRENDING, not just their
    # level.
    for col in ["gr_price", "ro_price"]:
        if col in out.columns:
            out[f"{col}_lag_48h"] = out[col].shift(48)
    if "ro_price" in out.columns:
        out["ro_price_lag_168h"] = out["ro_price"].shift(168)
    for col in ["gas_ttf", "coal_api2"]:
        if col in out.columns:
            out[f"{col}_lag_48h"] = out[col].shift(48)
            # value 48 h ago minus value a week earlier than that = weekly momentum
            out[f"{col}_chg_1w"]  = out[col].shift(48) - out[col].shift(48 + 168)

    return out


def feature_sets(feat):
    """Which columns the model uses — only those actually present in `feat`."""
    present = lambda cols: [c for c in cols if c in feat.columns]
    core = present(config.PRICE_LAGS + config.ROLLING + config.RAMP + config.DRIVERS
                   + config.WEATHER + config.NEIGHBOUR)
    fuel = present(config.FUEL)
    return {"core": core, "fuel": fuel,
            "tree": config.CAL_RAW + core + fuel,
            "linear": config.CAL_CYC + core + fuel}


def modeling_frame(feat):
    """Rows usable for training: every CORE feature and the target present.

    Missing fuels are kept (they only start ~2020 on Yahoo) — LightGBM reads NaN.
    """
    sets = feature_sets(feat)
    cols = list(dict.fromkeys(sets["tree"] + sets["linear"])) + [config.TARGET]
    return feat[cols].dropna(subset=sets["core"] + [config.TARGET])
