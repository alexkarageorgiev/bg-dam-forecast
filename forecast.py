"""48-hour forecast of the Bulgarian day-ahead electricity price (EUR/MWh).

    python forecast.py predict --demo          # offline replay of a recent day, no key needed
    python forecast.py predict --date 2026-06-15   # replay any past day in the dataset
    python forecast.py predict                 # live forecast for tomorrow + the day after (key)
    python forecast.py train                   # retrain on the cached dataset
    python forecast.py train --refresh         # re-download 2017 -> today first (key)

A forecast issued on day D covers the 48 hours D+1 00:00 -> D+2 23:00. Results are
printed and saved to outputs/forecast_<D+1>.csv and .png.
"""
import argparse
from datetime import datetime

import numpy as np
import pandas as pd

from dam_forecast import config, features, model, sources

HISTORY_DAYS = 40   # the 30-day rolling mean is shifted 48 h; fuels need 48 h + 1 week


# ── Building the 48 target rows ──────────────────────────────────────────────
def target_window(day):
    """Every hour of the two calendar days after `day` (48 h; 47 or 49 on DST days)."""
    start = day + pd.DateOffset(days=1)
    return pd.date_range(start, start + pd.DateOffset(days=2), freq="1h", inclusive="left")


def fill_missing_drivers(frame, window):
    """Day-ahead load/wind/solar forecasts are published once a day, so D+2 is never
    out yet. Any target hour still missing one takes the same hour a week earlier.
    Returns the patched frame and a per-hour flag of where the drivers came from."""
    frame = frame.copy()
    cols = [c for c in config.DRIVER_RAW if c in frame.columns]
    missing = frame.loc[window, cols].isna().any(axis=1)
    for c in cols:
        last_week = frame[c].shift(168)
        frame.loc[window, c] = frame.loc[window, c].fillna(last_week.loc[window])
    renew = [c for c in ["solar_fc_mw", "wind_onshore_fc_mw", "wind_offshore_fc_mw"]
             if c in frame.columns]
    frame.loc[window, "net_load_mw"] = (frame.loc[window, "load_fc_mw"]
                                        - frame.loc[window, renew].sum(axis=1))
    return frame, np.where(missing, "fallback_7d", "entsoe")


def as_known_at(frame, day):
    """Hide everything that would not be published at 08:00 on `day`:
    prices after D 23:00 and the D+2 driver forecasts."""
    frame = frame.copy()
    after_today = frame.index >= day + pd.DateOffset(days=1)
    frame.loc[after_today, [config.TARGET, *config.NEIGHBOURS]] = np.nan
    d2 = frame.index >= day + pd.DateOffset(days=2)
    frame.loc[d2, [c for c in config.DRIVER_RAW if c in frame.columns]] = np.nan
    return frame


# ── Commands ─────────────────────────────────────────────────────────────────
def cmd_train(args):
    df = sources.build_dataset() if args.refresh else sources.load_dataset()
    feat = features.make_features(df)
    mdf = features.modeling_frame(feat)
    feats = features.feature_sets(feat)["tree"]

    train, test = model.chrono_split(mdf)
    print(f"Holdout: {test.index.min():%Y-%m-%d} -> {test.index.max():%Y-%m-%d} "
          f"({len(test):,} hours never seen in training)")
    results = [model.evaluate("Seasonal naive (same hour last week)",
                              test[config.TARGET], test["price_lag_168h"]),
               model.evaluate("LightGBM", test[config.TARGET],
                              model.predict(model.fit(train, feats), test, feats))]
    print(model.scoreboard(results).to_string(), "\n")

    print(f"Refitting on all {len(mdf):,} hours ...")
    model.save(model.fit(mdf, feats), {
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "data_from": f"{mdf.index.min():%Y-%m-%d}", "data_to": f"{mdf.index.max():%Y-%m-%d}",
        "target_mode": config.TARGET_MODE, "features": feats,
        "holdout": {"from": f"{test.index.min():%Y-%m-%d}", "to": f"{test.index.max():%Y-%m-%d}",
                    "metrics": results}})
    print(f"Saved {config.MODEL_FILE.relative_to(config.ROOT)} and "
          f"{config.META_FILE.relative_to(config.ROOT)}")


def cmd_predict(args):
    if args.demo or args.date:
        forecast, history = replay(args)
    else:
        forecast, history = live()
    report(forecast, history)


def replay(args):
    """Re-run a past day from the cached dataset, exactly as if it were 08:00 on D.
    The model is refit on data up to D only (2 s), so it has never seen the answer."""
    df = sources.load_dataset()
    last_full = df[config.TARGET].dropna().index.max().normalize()
    day = (pd.Timestamp(args.date, tz=config.TZ) if args.date
           else last_full - pd.Timedelta(days=2))
    window = target_window(day)
    if window[-1] > df.index.max() or window[0] < config.HISTORY_START + pd.Timedelta(days=HISTORY_DAYS):
        raise SystemExit(f"--date must be between 2017-02-10 and {last_full - pd.Timedelta(days=2):%Y-%m-%d}")
    print(f"Replaying a forecast issued at 08:00 on {day:%a %d %b %Y} (offline, cached data)")

    actual = df.loc[window, config.TARGET]
    frame, source = fill_missing_drivers(as_known_at(df, day), window)
    feat = features.make_features(frame)
    feats = features.feature_sets(feat)["tree"]
    train = features.modeling_frame(feat.loc[feat.index < window[0]])
    print(f"Training LightGBM on {len(train):,} hours up to {train.index.max():%Y-%m-%d %H:%M} ...")
    m = model.fit(train, feats)
    pred = model.predict(m, feat.loc[window], feats)
    forecast = pd.DataFrame({"price_eur_mwh": pred, "actual_eur_mwh": actual.to_numpy(),
                             "drivers": source}, index=window)
    past_week = df[config.TARGET].loc[window[0] - pd.Timedelta(days=7):window[0] - pd.Timedelta(hours=1)]
    return forecast, past_week


def live():
    """Fetch the last 40 days + the next 2 from the APIs and forecast D+1 and D+2."""
    m, meta = model.load()
    day = pd.Timestamp.now(tz=config.TZ).normalize()
    window = target_window(day)
    print(f"Live forecast issued {pd.Timestamp.now(tz=config.TZ):%a %d %b %Y %H:%M} "
          f"(model trained on data to {meta['data_to']})")
    frame = sources.build_frame(day - pd.Timedelta(days=HISTORY_DAYS), window[-1] + pd.Timedelta(hours=1),
                                weather_mode="forecast")
    frame, source = fill_missing_drivers(as_known_at(frame, day), window)
    feat = features.make_features(frame)
    gaps = feat.loc[window, meta["features"]].isna().sum()
    if gaps.any():
        print("  note: missing inputs (LightGBM handles NaN, but check):",
              gaps[gaps > 0].to_dict())
    forecast = pd.DataFrame({"price_eur_mwh": model.predict(m, feat.loc[window], meta["features"],
                                                            meta["target_mode"]),
                             "drivers": source}, index=window)
    return forecast, frame[config.TARGET].loc[:window[0]].dropna().iloc[-24 * 7:]


# ── Output ───────────────────────────────────────────────────────────────────
def report(fc, history):
    days = [g for _, g in fc.groupby(fc.index.date)]
    cols = {}
    for g in days:
        labels = pd.Series(g.index.strftime("%H:%M"))
        labels[labels.duplicated()] += "*"          # the repeated hour when DST ends
        by_hour = g.set_index(pd.Index(labels))
        cols[f"{g.index[0]:%a %d %b}"] = by_hour["price_eur_mwh"]
        if "actual_eur_mwh" in g:
            cols[f"actual {g.index[0]:%d %b}"] = by_hour["actual_eur_mwh"]
    table = pd.DataFrame(cols).sort_index().rename_axis("hour")
    print("\nForecast, EUR/MWh   (drivers: " + ", ".join(
        f"{g.index[0]:%d %b} = {g['drivers'].mode()[0]}" for g in days) + ")")
    print(table.round(1).to_string(na_rep="-"))

    for g in days:
        p = g["price_eur_mwh"]
        print(f"{g.index[0]:%a %d %b}: mean {p.mean():6.1f} | cheapest {p.idxmin():%H:%M} "
              f"({p.min():.1f}) | most expensive {p.idxmax():%H:%M} ({p.max():.1f})")
    if "actual_eur_mwh" in fc:
        err = (fc["price_eur_mwh"] - fc["actual_eur_mwh"]).abs().mean()
        last_week = history.reindex(fc.index - pd.Timedelta(days=7)).to_numpy()
        naive = np.nanmean(np.abs(last_week - fc["actual_eur_mwh"].to_numpy()))
        print(f"MAE vs actual: {err:.1f} EUR/MWh   (seasonal-naive 'same hour last week': {naive:.1f})")

    config.OUTPUT_DIR.mkdir(exist_ok=True)
    stem = config.OUTPUT_DIR / f"forecast_{fc.index[0]:%Y-%m-%d}"
    fc.rename_axis("time").to_csv(f"{stem}.csv", float_format="%.2f")
    plot(fc, history, f"{stem}.png")
    print(f"\nSaved {stem.relative_to(config.ROOT)}.csv and .png")


def plot(fc, history, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    naive = lambda idx: idx.tz_localize(None)
    fig, ax = plt.subplots(figsize=(12, 4.2), dpi=110)
    ax.plot(naive(history.index), history, color="0.35", lw=1.3, label="actual (past week)")
    if "actual_eur_mwh" in fc:
        ax.plot(naive(fc.index), fc["actual_eur_mwh"], color="black", lw=1.6, label="actual")
    ax.plot(naive(fc.index), fc["price_eur_mwh"], color="#d6604d", lw=2.2, label="forecast")
    fb = fc["drivers"] == "fallback_7d"
    if fb.any():
        ax.axvspan(naive(fc.index[fb])[0], naive(fc.index[fb])[-1], color="#d6604d", alpha=0.07,
                   label="drivers = last week's (D+2)")
    ax.axvline(naive(fc.index[:1])[0], color="0.6", ls="--", lw=1)
    ax.axhline(0, color="0.8", lw=0.8)
    ax.set_ylabel("EUR/MWh")
    ax.set_title(f"BG day-ahead price — 48 h forecast for {fc.index[0]:%d %b} and "
                 f"{fc.index[-1]:%d %b %Y}", loc="left")
    ax.legend(loc="upper left", ncol=4, fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("predict", help="forecast the next 48 hours (or replay a past day)")
    p.add_argument("--demo", action="store_true", help="replay the most recent day in the dataset, offline")
    p.add_argument("--date", help="replay the forecast issued on this past day (YYYY-MM-DD), offline")
    p.set_defaults(func=cmd_predict)
    t = sub.add_parser("train", help="retrain the model and save it to models/")
    t.add_argument("--refresh", action="store_true", help="re-download the dataset first (needs a key)")
    t.set_defaults(func=cmd_train)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
