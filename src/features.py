"""Shared feature engineering. All modelling scripts must build features through here
so that results across workstreams are comparable.

Convention: a row is (station, origin time t). The target is y(t+h) for horizon h.
Lag features only use information available at t. Calendar / weather / event features
describe the *target* hour t+h (weather: observed by default, or forecast if supplied).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import EXTERNAL, PANEL_HIST_PATH, PANEL_PATH, TRAIN_END  # noqa: E402


def load_panel(hist: bool = False) -> pd.DataFrame:
    p = pd.read_parquet(PANEL_HIST_PATH if hist else PANEL_PATH)
    return p.sort_values(["station", "ts"], kind="stable").reset_index(drop=True)


def load_calendar() -> pd.DataFrame:
    cal = pd.read_csv(EXTERNAL / "calendar.csv", parse_dates=["date"])
    return cal


def load_weather(kind: str = "obs") -> pd.DataFrame | None:
    """Hourly Taipei weather. kind='obs' -> weather_hourly.csv, 'forecast' -> weather_forecast_hourly.csv.
    Expected columns: ts, temp, precip, rh (+ optional extras). Returns None if not yet available."""
    f = EXTERNAL / ("weather_hourly.csv" if kind == "obs" else "weather_forecast_hourly.csv")
    if not f.exists():
        return None
    w = pd.read_csv(f, parse_dates=["ts"])
    return w


def load_events() -> pd.DataFrame | None:
    """Large events. Expected columns of external/events_station_hour.csv: ts, station, event_flag,
    event_scale (optional). Returns None if not yet available."""
    f = EXTERNAL / "events_station_hour.csv"
    if not f.exists():
        return None
    return pd.read_csv(f, parse_dates=["ts"])


def _g_shift(s: pd.Series, n: int, key: np.ndarray) -> np.ndarray:
    """Shift within station for a panel already sorted by (station, ts) with equal length blocks."""
    v = s.to_numpy(dtype="float64")
    out = np.full_like(v, np.nan)
    if n > 0:
        out[n:] = v[:-n]
        same = np.zeros_like(v, dtype=bool)
        same[n:] = key[n:] == key[:-n]
    else:
        out[:n] = v[-n:]
        same = np.zeros_like(v, dtype=bool)
        same[:n] = key[:n] == key[-n:]
    out[~same] = np.nan
    return out


def _g_roll_mean(s: pd.Series, w: int, by: pd.Series) -> np.ndarray:
    return s.groupby(by, observed=True).transform(lambda x: x.rolling(w, min_periods=w // 2).mean()).to_numpy()


def make_features(panel: pd.DataFrame, target: str = "entries", horizon: int = 1,
                  weather: pd.DataFrame | None = "auto", events: pd.DataFrame | None = "auto",
                  extra_lags: bool = True, train_end: str = TRAIN_END) -> pd.DataFrame:
    """Return a feature frame with columns: ts (origin), ts_target, station, y, features..., scale.

    Features (proposal 3.2): target-hour hour/dow/holiday, station, lags 1/2/3 at origin,
    seasonal lags 24/168 relative to target, rolling 24/168 means, historical average
    (station x day_type x hour, from TRAIN only), weather at target hour, other-direction lag.
    `scale` = station mean over previous 168h (used by the scaled model).
    `train_end` bounds the window used for hist_avg (override for rolling-origin backtests)."""
    df = panel[["ts", "station", "entries", "exits"]].copy()
    other = "exits" if target == "entries" else "entries"
    key = df["station"].cat.codes.to_numpy()
    y = df[target]
    h = horizon

    out = pd.DataFrame({"ts": df["ts"], "station": df["station"]})
    out["ts_target"] = df["ts"] + pd.Timedelta(hours=h)
    out["y"] = _g_shift(y, -h, key)

    for L in (0, 1, 2):                                  # y_t, y_{t-1}, y_{t-2}  ("lag1..3" of target)
        out[f"lag_{L + h}"] = _g_shift(y, L, key) if L else y.to_numpy(dtype="float64")
    out["lag_24"] = _g_shift(y, 24 - h, key)              # same hour yesterday (relative to target)
    out["lag_168"] = _g_shift(y, 168 - h, key)            # same hour last week
    if extra_lags:
        out["lag_336"] = _g_shift(y, 336 - h, key)
        out["lag_48"] = _g_shift(y, 48 - h, key)
        out[f"other_lag_{h}"] = df[other].to_numpy(dtype="float64")
        out["diff_1"] = out[f"lag_{h}"] - out[f"lag_{h + 1}"]
    out["roll_24"] = _g_roll_mean(y, 24, df["station"])
    out["roll_168"] = _g_roll_mean(y, 168, df["station"])
    out["scale"] = out["roll_168"].clip(lower=1.0)

    # calendar at target hour
    tt = out["ts_target"]
    out["hour"] = tt.dt.hour.astype("int8")
    out["dow"] = tt.dt.dayofweek.astype("int8")
    out["month"] = tt.dt.month.astype("int8")
    cal = load_calendar().set_index("date")
    d = tt.dt.normalize()
    for c in ("is_holiday", "is_makeup_workday", "is_long_weekend", "holiday_block_len",
              "days_to_next_holiday", "days_since_last_holiday"):
        out[c] = d.map(cal[c]).fillna(0).astype("int8")

    # historical average: station x day_type x hour on TRAIN period only (no leakage)
    hist = pd.DataFrame({"station": df["station"], "hour": df["ts"].dt.hour,
                         "hol": df["ts"].dt.normalize().map(cal["is_holiday"]).fillna(False).astype(int),
                         "v": y})
    hist = hist[df["ts"] <= train_end]
    ha = hist.groupby(["station", "hol", "hour"], observed=True)["v"].mean().rename("hist_avg")
    ha = ha.reset_index().rename(columns={"hol": "is_holiday"}).astype({"is_holiday": "int8", "hour": "int8"})
    out = out.merge(ha, on=["station", "is_holiday", "hour"], how="left")

    if isinstance(weather, str) and weather == "auto":
        weather = load_weather("obs")
    if weather is not None:
        wcols = [c for c in weather.columns if c != "ts"]
        out = out.merge(weather.rename(columns={"ts": "ts_target"}), on="ts_target", how="left")
        if "precip" in wcols:
            out["precip_3h"] = out.groupby("station", observed=True)["precip"].transform(
                lambda x: x.rolling(3, min_periods=1).sum())
    if isinstance(events, str) and events == "auto":
        events = load_events()
    if events is not None:
        ev = events.rename(columns={"ts": "ts_target"})
        ev["station"] = pd.Categorical(ev["station"], categories=out["station"].cat.categories)
        out = out.merge(ev, on=["ts_target", "station"], how="left")
        for c in ev.columns:
            if c not in ("ts_target", "station"):
                out[c] = out[c].fillna(0)
    out["station"] = out["station"].astype(df["station"].dtype)
    return out


BASE_EXCLUDE = {"ts", "ts_target", "y", "scale"}


def feature_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if c not in BASE_EXCLUDE]
