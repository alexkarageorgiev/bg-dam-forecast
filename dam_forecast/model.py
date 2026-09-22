"""The production model: LightGBM, scored on a chronological holdout.

Same recipe as notebooks/02_model.ipynb, "Models":
  * a chronological split, never shuffled — shuffling lets the model peek at the future;
  * target: the price itself ("direct"). The research also tried fitting price MINUS
    its 30-day level ("residual") — the notebook measures both and direct wins,
    so config.TARGET_MODE = "direct"; the residual path stays for comparison;
  * scored with MAE / RMSE / sMAPE (not MAPE: DAM prices touch zero and go negative).
"""
import json

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from . import config


def chrono_split(model_df, test_fraction=config.TEST_FRACTION):
    cut = int(len(model_df) * (1 - test_fraction))
    return model_df.iloc[:cut], model_df.iloc[cut:]


def _baseline(frame, mode):
    return frame[config.BASELINE_COL].to_numpy() if mode == "residual" else 0.0


def fit(train, features, mode=config.TARGET_MODE):
    y = train[config.TARGET] - _baseline(train, mode)
    return LGBMRegressor(**config.LGBM_PARAMS).fit(train[features], y)


def predict(model, frame, features, mode=config.TARGET_MODE):
    return model.predict(frame[features]) + _baseline(frame, mode)


def evaluate(name, y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    err = y_pred - y_true
    smape = (2 * np.abs(err) / (np.abs(y_true) + np.abs(y_pred) + 1e-9)).mean() * 100
    return {"model": name,
            "MAE": float(np.abs(err).mean()),
            "RMSE": float(np.sqrt((err ** 2).mean())),
            "sMAPE_%": float(smape)}


def save(model, meta):
    config.MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, config.MODEL_FILE)
    config.META_FILE.write_text(json.dumps(meta, indent=2))


def load():
    if not config.MODEL_FILE.exists():
        raise SystemExit("No trained model in models/. Run:  python forecast.py train")
    return joblib.load(config.MODEL_FILE), json.loads(config.META_FILE.read_text())


def scoreboard(results):
    return pd.DataFrame(results).set_index("model").round(2)
