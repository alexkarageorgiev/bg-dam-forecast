# Bulgarian day-ahead electricity price — 48 h forecast

Forecasts the hourly price on Bulgaria's day-ahead market (IBEX, EUR/MWh) for tomorrow and the day after,
from public data only: ENTSO-E grid forecasts, Open-Meteo weather, neighbouring markets and fuel prices.
Two research notebooks explain how the model was chosen. `forecast.py` runs it on live data.

![48-hour forecast replayed for 20–21 Sep 2026 against the actual price](outputs/example_forecast.png)

*Replay of the forecast made at 08:00 on 19 Sep 2026. The model had never seen these two days.
Output of `python forecast.py predict --demo`.*

## Results

Holdout: the most recent 15 % of the data (6 Apr 2025 → 21 Sep 2026, 12,381 hours), split by time and never shuffled.

| Model | MAE | RMSE | sMAPE |
|---|---:|---:|---:|
| Seasonal naive — *same hour last week* | 28.6 | 43.0 | 44.3 % |
| LSTM (168 h lookback) | 27.9 | 37.1 | 40.6 % |
| Ridge regression | 25.2 | 34.7 | 40.8 % |
| **LightGBM** (shipped) | **23.3** | **33.0** | **36.7 %** |

EUR/MWh. LightGBM cuts the benchmark error by 18 %. sMAPE is reported instead of MAPE because prices reach
zero and go negative. Every number comes from [`02_model.ipynb`](notebooks/02_model.ipynb), which also checks
that it matches the shipped model's [`models/meta.json`](models/meta.json).

## Quickstart

No API key needed. Python 3.12.

```bash
git clone https://github.com/alexkarageorgiev/bg-dam-forecast.git && cd bg-dam-forecast
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python forecast.py predict --demo    # replay a recent day offline: table + chart in outputs/
jupyter lab notebooks/               # the research, top to bottom
```

`python forecast.py predict --date 2026-06-15` replays any day since Feb 2017, with the model retrained on
data up to that day only. `python forecast.py --help` lists every command.

## Reading order

| | What you'll find |
|---|---|
| 1. [`notebooks/01_exploration.ipynb`](notebooks/01_exploration.ipynb) | One year of prices: solar collapse, evening peak, negative prices, the temperature sign flip, and the benchmark to beat. |
| 2. [`notebooks/02_model.ipynb`](notebooks/02_model.ipynb) | Nine years of data, the features, and five models compared on the same holdout: where LightGBM wins and where it still misses. |
| 3. [`dam_forecast/`](dam_forecast) | The code both notebooks and the CLI share: [`sources.py`](dam_forecast/sources.py) fetches, [`features.py`](dam_forecast/features.py) builds features, [`model.py`](dam_forecast/model.py) trains and scores. |
| 4. [`forecast.py`](forecast.py) | The result: a 48-hour forecast from live data. |

GitHub shows the notebooks with static figures. For the interactive Plotly versions, open them on
[nbviewer](https://nbviewer.org) or run them locally.

## How it works

**No look-ahead.** The forecast is made in the morning of day D for D+1 and D+2, so a feature may only use
what is published by then.
- Day-ahead load, wind and solar forecasts and the weather forecast are used at their own hour.
- Anything price-derived (the BG price, Greek and Romanian prices, gas) enters only as a lag of **at least 48 hours**.
- Rolling windows are shifted 48 h before they roll.
- Fuel closes are stamped on the following day, because a close settles in the evening.

| Source | Data | Key |
|---|---|---|
| [ENTSO-E Transparency](https://transparency.entsoe.eu) | BG price (target); day-ahead load, wind and solar forecasts; GR and RO prices | free token |
| [Open-Meteo](https://open-meteo.com) | Sofia temperature, wind, radiation, cloud: archive for training, forecast for live runs | none |
| Yahoo Finance | TTF gas (used), Brent and KRBN carbon proxy (fetched) | none |

**Why LightGBM, not an LSTM.** The price drivers are already published as day-ahead forecasts. That makes this
regression with known inputs rather than blind sequence extrapolation, and gradient-boosted trees are strong
at that. The LSTM was tested on the same split and lost to both LightGBM and Ridge.

**What the research changed.** The first design predicted price *minus its 30-day average*. Measured on the
holdout, that was worse (MAE 27.2) and fell behind the naive benchmark in 2026 Q3. The shipped model therefore
predicts the price directly. Coal was dropped because Yahoo stopped publishing API2 coal in Dec 2025, so a live
forecast could never have it.

## Live forecasts

1. Register for free at [transparency.entsoe.eu](https://transparency.entsoe.eu), then request API access
   (email transparency@entsoe.eu, subject "Restful API access") and generate a token under *My Account Settings*.
2. `cp .env.example .env` and paste the token.
3. Run:

```bash
python forecast.py predict          # tomorrow + the day after, from today's data
python forecast.py train --refresh  # re-download 2017 -> today and retrain (~5 min)
```

Results are printed and saved as `outputs/forecast_<date>.csv` and `.png`. Each hour is flagged
`drivers = entsoe | fallback_7d`, which shows where the grid forecasts were still unpublished.

## Limitations

- **D+2 inputs are estimated.** ENTSO-E publishes grid forecasts one day ahead, so D+2 uses the same hour last
  week (`fallback_7d`). Replays apply the same rule.
- **Weather differs between training and live runs.** Training uses reanalysis (observed weather); live runs
  use the forecast. Replays therefore see slightly better weather than a real run would.
- **Evening peaks and spikes are under-called.** MAE is 39 at 20:00 and ~53 on hours above 200 EUR/MWh.
- **Carbon has no clean free source.** KRBN is an ETF proxy for the EU ETS price. It is fetched but not used.

## Roadmap

- The v1 ideas from the exploration notebook: hour × month interactions, a classifier for price ≤ 0 hours,
  and temperature per hour block.
- A daily run on a Raspberry Pi (cron), feeding a Blynk dashboard.

## Repository layout

```
├── forecast.py            CLI: predict (--demo | --date) and train
├── dam_forecast/          config, sources, features, model — shared by notebooks and CLI
├── notebooks/             01_exploration, 02_model (committed with outputs)
├── data/                  cached datasets, so everything runs without a key
├── models/                the trained LightGBM + meta.json (training range, holdout scores)
└── outputs/               forecast CSV + PNG land here
```

Built as an internship project. MIT licence.
