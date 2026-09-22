"""Data sources -> one hourly, tz-aware Europe/Sofia frame.

ENTSO-E    BG day-ahead price (target); day-ahead load, wind & solar forecasts;
           GR and RO day-ahead prices                       (needs a free API key)
Open-Meteo Sofia weather — archive for history, forecast for live runs  (no key)
Yahoo      TTF gas, API2 coal, Brent, KRBN carbon proxy — daily closes (no key)

Every series is tz-aware: mixing tz-aware and tz-naive indexes makes pd.concat
misalign and silently fill columns with NaN.
"""
import os

import pandas as pd
import requests

from . import config
from .features import standardize


def entsoe_client():
    """ENTSO-E client using ENTSOE_API_KEY from the environment or ./.env."""
    from dotenv import load_dotenv
    from entsoe import EntsoePandasClient

    load_dotenv(config.ROOT / ".env")
    key = os.environ.get("ENTSOE_API_KEY", "").strip()
    if not key or key.startswith("your_"):
        raise SystemExit(
            "No ENTSO-E API key found.\n"
            "  1. Register (free) at https://transparency.entsoe.eu and request an API token.\n"
            "  2. cp .env.example .env   and paste the token after ENTSOE_API_KEY=\n"
            "Everything except live data runs without a key: try  python forecast.py predict --demo")
    return EntsoePandasClient(api_key=key)


def fetch_entsoe(client, start, end):
    """BG price + day-ahead load/wind/solar forecasts + neighbour prices, raw names."""
    parts = [client.query_day_ahead_prices(config.ZONE, start=start, end=end)
             .rename(config.TARGET)]
    for query in (client.query_load_forecast, client.query_wind_and_solar_forecast):
        try:
            parts.append(query(config.ZONE, start=start, end=end))
        except Exception as e:           # e.g. forecasts for D+2 are not published yet
            print(f"  ENTSO-E {query.__name__}: {type(e).__name__} — left empty")
    for col, zone in config.NEIGHBOURS.items():
        try:
            parts.append(client.query_day_ahead_prices(zone, start=start, end=end).rename(col))
        except Exception as e:
            print(f"  ENTSO-E {zone} price: {type(e).__name__} — left empty")
    return pd.concat(parts, axis=1)


def fetch_weather(start, end, mode="archive"):
    """Hourly Sofia weather. mode="archive" (reanalysis, history) or "forecast" (live)."""
    url = {"archive": "https://archive-api.open-meteo.com/v1/archive",
           "forecast": "https://api.open-meteo.com/v1/forecast"}[mode]
    pad = pd.Timedelta(days=1)           # the UTC -> Sofia shift must not clip the window
    last = end + pad
    if mode == "archive":                # the archive refuses dates in the future
        last = min(last, pd.Timestamp.now(tz=config.TZ).normalize())
    r = requests.get(url, timeout=120, params={
        "latitude": config.LAT, "longitude": config.LON,
        "start_date": (start - pad).strftime("%Y-%m-%d"),
        "end_date": last.strftime("%Y-%m-%d"),
        "hourly": ",".join(config.WEATHER_VARS), "timezone": "UTC"})
    r.raise_for_status()
    w = pd.DataFrame(r.json()["hourly"])
    w["time"] = pd.to_datetime(w["time"]).dt.tz_localize("UTC").dt.tz_convert(config.TZ)
    return w.set_index("time").sort_index()


def fetch_commodities(start, end):
    """Daily closes, stamped at 00:00 of the NEXT day.

    A close settles in the evening, so it is only knowable from the following
    morning. Stamping it on its own date would leak ~18 h of the future.
    """
    import yfinance as yf

    out = {}
    for col, ticker in config.COMMODITIES.items():
        try:
            d = yf.download(ticker, start=start.strftime("%Y-%m-%d"),
                            end=end.strftime("%Y-%m-%d"), progress=False, auto_adjust=True)
            s = d["Close"].squeeze().dropna()
            s.index = pd.to_datetime(s.index).tz_localize(config.TZ) + pd.Timedelta(days=1)
            out[col] = s
        except Exception as e:
            print(f"  Yahoo {ticker}: {type(e).__name__} — left empty")
    return out


def build_frame(start, end, weather_mode="archive", client=None):
    """Fetch every source for [start, end) and return the standardized hourly frame."""
    client = client or entsoe_client()
    print(f"Fetching ENTSO-E {start:%Y-%m-%d} -> {end:%Y-%m-%d} ...")
    raw = fetch_entsoe(client, start, end)
    print("Fetching Open-Meteo weather ...")
    weather = fetch_weather(start, end, weather_mode)
    recent = pd.Timestamp.now(tz=config.TZ) - pd.Timedelta(days=10)
    if weather_mode == "archive" and end > recent:
        # The archive lags real time by ~5 days; fill the gap from the forecast API.
        weather = weather.combine_first(fetch_weather(max(start, recent), end, "forecast"))
    df = standardize(pd.concat([raw, weather], axis=1))
    grid = pd.date_range(start, end, freq="1h", inclusive="left")
    df = df.reindex(grid)
    print("Fetching Yahoo Finance fuel & carbon closes ...")
    for col, s in fetch_commodities(start - pd.Timedelta(days=14), end).items():
        # Carry a close over weekends and holidays, but not further: a ticker that
        # stops updating (Yahoo's API2 coal did in Dec 2025) must become NaN, not a
        # frozen price. LightGBM reads NaN natively.
        df[col] = s.reindex(grid.union(s.index)).ffill(limit=24 * 5).reindex(grid)
    return df


def load_dataset():
    """The cached training frame shipped with the repo (no key needed)."""
    return pd.read_parquet(config.DATASET)


def build_dataset(end=None):
    """Re-download the full 2017 -> `end` history and overwrite the cache."""
    end = end or pd.Timestamp.now(tz=config.TZ).normalize()
    df = build_frame(config.HISTORY_START, end, weather_mode="archive")
    config.DATA_DIR.mkdir(exist_ok=True)
    df.to_parquet(config.DATASET)
    print(f"Saved {len(df):,} hours to {config.DATASET.relative_to(config.ROOT)}")
    return df
