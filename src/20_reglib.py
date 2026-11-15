"""Workstream B shared helpers (regression). Imported by the other src/20_*.py scripts via
`reglib = importlib.import_module("20_reglib")`-style loading (see `load()` at the bottom of each script).

Builds on the shared `features.make_features` and adds B-specific derived features:
  * station-level normalised lags (``*_s`` = value / scale) so one tree structure fits all station sizes;
  * historical averages recomputed with an arbitrary cut-off (no leakage in rolling-origin backtests),
    by station x day-type(workday/Sat/Sun-holiday) x hour, and the same profile at the *origin* hour so the
    model sees the current anomaly (y_t - profile_t);
  * weather derivatives (rain onset, change, previous-3h accumulation, rain x holiday) when weather exists.
"""
from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import (EXTERNAL, RAIN_MM, TEST_END, TEST_START, TRAIN_END, VALID_END,  # noqa: E402,F401
                    VALID_START)
from features import feature_columns, load_calendar, load_events, load_weather, make_features  # noqa: E402

LGB_PARAMS = dict(learning_rate=0.05, num_leaves=127, min_data_in_leaf=100, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=0, seed=42)

# objective of LightGBM-B (L2 on y/scale with weight=scale). L1 was tried: worse on validation and ~8x slower.
B_OBJ = "regression"
B_PARAMS: dict = {}

# columns of the progress-report "first version" model (same as src/04_baselines_t1.py)
V1_DROP = {"hist_avg", "lag_336", "lag_48", "diff_1", "is_makeup_workday", "is_long_weekend",
           "holiday_block_len", "days_to_next_holiday", "days_since_last_holiday", "month"}
ALWAYS_DROP = {"n_points"}  # bookkeeping columns of the weather file, not features
B_ONLY = set()  # filled by add_features (names of the B-specific columns)


def _daytype(ts: pd.Series, cal: pd.DataFrame) -> np.ndarray:
    """0 = workday, 1 = Saturday (non-holiday-block), 2 = Sunday / holiday."""
    d = ts.dt.normalize()
    hol = d.map(cal["is_holiday"]).fillna(False).astype(bool).to_numpy()
    mk = d.map(cal["is_makeup_workday"]).fillna(False).astype(bool).to_numpy()
    dow = ts.dt.dayofweek.to_numpy()
    out = np.zeros(len(ts), dtype="int8")
    out[(dow == 5) & ~mk] = 1
    out[(dow == 6) & ~mk] = 2
    out[hol] = 2
    out[mk] = 0
    return out


def profile_table(panel: pd.DataFrame, target: str, hist_end, hist_start=None) -> pd.DataFrame:
    """Mean of target by station x hol(0/1) x hour and station x daytype x hour using ts <= hist_end."""
    cal = load_calendar().set_index("date")
    m = panel["ts"] <= pd.Timestamp(hist_end)
    if hist_start is not None:
        m &= panel["ts"] >= pd.Timestamp(hist_start)
    p = panel.loc[m]
    df = pd.DataFrame({"station": p["station"], "hour": p["ts"].dt.hour.astype("int8"),
                       "is_holiday": p["ts"].dt.normalize().map(cal["is_holiday"]).fillna(False).astype("int8"),
                       "daytype": _daytype(p["ts"], cal), "v": p[target].astype(float)})
    ha = df.groupby(["station", "is_holiday", "hour"], observed=True)["v"].mean().rename("hist_avg").reset_index()
    hd = df.groupby(["station", "daytype", "hour"], observed=True)["v"].mean().rename("hist_dt").reset_index()
    return ha, hd


def add_features(fr: pd.DataFrame, panel: pd.DataFrame, target: str, h: int, hist_end=TRAIN_END,
                 hist_start=None) -> pd.DataFrame:
    """Recompute hist_avg with `hist_end` and add B features. Returns a new frame (same row order)."""
    cal = load_calendar().set_index("date")
    ha, hd = profile_table(panel, target, hist_end, hist_start)
    fr = fr.drop(columns=["hist_avg"]).reset_index(drop=True)
    n = len(fr)
    fr["daytype"] = _daytype(fr["ts_target"], cal)
    fr = fr.merge(ha, on=["station", "is_holiday", "hour"], how="left")
    fr = fr.merge(hd, on=["station", "daytype", "hour"], how="left")
    # profile at origin hour t (to measure the current anomaly)
    o = pd.DataFrame({"station": fr["station"], "hour": fr["ts"].dt.hour.astype("int8"),
                      "daytype": _daytype(fr["ts"], cal)})
    o = o.merge(hd.rename(columns={"hist_dt": "hist_dt_origin"}), on=["station", "daytype", "hour"], how="left")
    fr["hist_dt_origin"] = o["hist_dt_origin"].to_numpy()
    assert len(fr) == n
    s = fr["scale"]
    lag_h = fr[f"lag_{h}"]
    fr["anom_origin_s"] = (lag_h - fr["hist_dt_origin"]) / s
    fr["ratio_origin"] = (lag_h + 10) / (fr["hist_dt_origin"] + 10)
    fr["ratio_168"] = (fr["lag_168"] + 10) / (fr["hist_dt"] + 10)
    for c in [f"lag_{h}", f"lag_{h + 1}", f"lag_{h + 2}", "lag_24", "lag_168", "lag_336", "hist_dt", "hist_avg",
              "roll_24"]:
        fr[f"{c}_s"] = fr[c] / s
    fr["log_scale"] = np.log1p(s)
    ty = typhoon_dates()
    if ty is not None:  # typhoon closures are announced the evening before -> known at prediction time
        fr["is_typhoon_day"] = fr["ts_target"].dt.normalize().isin(ty).astype("int8")
    if "precip" in fr:
        g = fr.groupby("station", observed=True)["precip"]
        fr["precip_prev1"] = g.shift(1)
        fr["precip_prev3"] = g.transform(lambda x: x.shift(1).rolling(3, min_periods=1).sum())
        fr["precip_6h"] = g.transform(lambda x: x.rolling(6, min_periods=1).sum())
        fr["precip_change"] = fr["precip"] - fr["precip_prev1"]
        fr["rain_onset"] = ((fr["precip"] >= RAIN_MM) & (fr["precip_prev1"].fillna(0) < RAIN_MM)).astype("int8")
        wet = (fr["precip"] >= RAIN_MM).astype(int)
        # consecutive rainy hours up to target hour
        grp = (wet != wet.groupby(fr["station"], observed=True).shift(1)).cumsum()
        fr["rain_run"] = wet.groupby([fr["station"], grp], observed=True).cumsum().astype("int16")
        fr["rain_x_holiday"] = fr["precip"] * (fr["daytype"] > 0)
        # rain observed at origin t (genuinely known at prediction time)
        fr["precip_origin"] = g.shift(h)
    B_ONLY.update(c for c in fr.columns if c.endswith("_s") or c in (
        "daytype", "is_typhoon_day", "hist_dt", "hist_dt_origin", "ratio_origin", "ratio_168", "log_scale", "precip_prev1",
        "precip_prev3", "precip_6h", "precip_change", "rain_onset", "rain_run", "rain_x_holiday", "precip_origin"))
    return fr


def typhoon_dates():
    """Typhoon work/school suspension days (workstream A: data/external/typhoon_days.csv) or None."""
    f = EXTERNAL / "typhoon_days.csv"
    if not f.exists():
        return None
    return pd.to_datetime(pd.read_csv(f)["date"]).dt.normalize()


def build(panel: pd.DataFrame, target: str, h: int, weather="auto", events="auto", hist_end=TRAIN_END,
          hist_start=None) -> pd.DataFrame:
    fr = make_features(panel, target=target, horizon=h, weather=weather, events=events)
    return add_features(fr, panel, target, h, hist_end, hist_start)


def split(fr: pd.DataFrame, train_end=TRAIN_END, valid=(VALID_START, VALID_END), test=(TEST_START, TEST_END),
          train_start=None):
    fr = fr.dropna(subset=["y", "lag_168"])
    t = fr["ts_target"]
    mtr = t <= pd.Timestamp(train_end)
    if train_start is not None:
        mtr &= t >= pd.Timestamp(train_start)
    return (fr[mtr], fr[(t >= valid[0]) & (t <= valid[1])], fr[(t >= test[0]) & (t <= test[1])])


def _fc(fr):
    return [c for c in feature_columns(fr) if c not in ALWAYS_DROP]


def cols_v1(fr):
    return [c for c in _fc(fr) if c not in V1_DROP and c not in B_ONLY and not c.startswith("other_lag")]


def cols_scaled(fr):
    """Feature set of the progress-report LightGBM-scaled model (shared features only)."""
    return [c for c in _fc(fr) if c not in B_ONLY]


def cols_full(fr, drop=()):
    return [c for c in _fc(fr) if c not in drop]


def fit(tr, va, cols, objective="regression", scaled=True, params=None, rounds=3000, es=100, alpha=None,
        weight_scaled=True):
    """Train LightGBM. scaled=True: target y/scale, weights scale (so the loss is on the original scale for
    L1/quantile, and approximately so for L2)."""
    if scaled:
        ytr, yva = tr.y / tr.scale, va.y / va.scale
        wtr, wva = (tr.scale, va.scale) if weight_scaled else (None, None)
    else:
        ytr, yva, wtr, wva = tr.y, va.y, None, None
    prm = {**LGB_PARAMS, **(params or {}), "objective": objective}
    if alpha is not None:
        prm["alpha"] = alpha
    cat = ["station"] if "station" in cols else []
    dtr = lgb.Dataset(tr[cols], ytr, weight=wtr, categorical_feature=cat, free_raw_data=True)
    dva = lgb.Dataset(va[cols], yva, weight=wva, reference=dtr)
    return lgb.train(prm, dtr, num_boost_round=rounds, valid_sets=[dva],
                     callbacks=[lgb.early_stopping(es, verbose=False)])


def predict(m, fr, cols, scaled=True, clip=True):
    p = m.predict(fr[cols], num_iteration=m.best_iteration)
    p = p * fr["scale"].to_numpy() if scaled else p
    return np.clip(p, 0, None) if clip else p


def pinball(y, q, a):
    d = np.asarray(y, float) - np.asarray(q, float)
    return float(np.mean(np.maximum(a * d, (a - 1) * d)))


def weather_available() -> bool:
    return (EXTERNAL / "weather_hourly.csv").exists()


def forecast_available() -> bool:
    return (EXTERNAL / "weather_forecast_hourly.csv").exists()


def md_table(df: pd.DataFrame, floatfmt="{:.1f}") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                cells.append("" if pd.isna(v) else floatfmt.format(v))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
