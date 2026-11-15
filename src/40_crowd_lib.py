"""Shared helpers for workstream D (spatial features + crowd warning).

Imported by src/40_crowd_baseline.py, 41_spatial_features.py, 42_crowd_main.py and
43_alert_backtest.py via ``importlib.import_module("40_crowd_lib")``.

Conventions
-----------
* Crowd label (proposal 3.3): total = entries + exits. A station-hour is "crowded" when
  total >= the station's P95 of total over the *training* period. The forecast row is
  (station, origin t) and the label refers to t+h.
* Note on the time index of the panel: exits are indexed by the *entry* hour of the trip
  (network-wide sum of entries == sum of exits every hour), because the open OD file only
  carries the entry hour. "total(t)" therefore counts trips that start at, or are bound
  for, the station and entered the network during hour t.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import (CROWD_QUANTILE, INTERIM, TEST_END, TEST_START, TRAIN_END,  # noqa: E402
                    VALID_END, VALID_START)
from features import feature_columns, make_features  # noqa: E402

LGB_REG = dict(learning_rate=0.05, num_leaves=127, min_data_in_leaf=100, feature_fraction=0.8,
               bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=0)
LGB_CLF = dict(learning_rate=0.05, num_leaves=63, min_data_in_leaf=100, feature_fraction=0.8,
               bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=0)
CACHE = INTERIM / "ws_d"
CACHE.mkdir(parents=True, exist_ok=True)


def mod(name: str):
    """Import a sibling script whose file name starts with a digit (e.g. '41_spatial_features')."""
    return importlib.import_module(name)


# --------------------------------------------------------------------------- labels
def crowd_thresholds(panel: pd.DataFrame, train_end: str = TRAIN_END, q: float = CROWD_QUANTILE,
                     train_start: str | None = None) -> pd.Series:
    """Per-station quantile of entries+exits on the training window."""
    m = panel["ts"] <= train_end
    if train_start is not None:
        m &= panel["ts"] >= train_start
    tot = (panel.loc[m, "entries"] + panel.loc[m, "exits"]).rename("total")
    thr = tot.groupby(panel.loc[m, "station"], observed=True).quantile(q)
    return thr.rename("crowd_thr")


def abs_threshold(panel: pd.DataFrame, train_end: str = TRAIN_END, q: float = 0.95,
                  train_start: str | None = None) -> float:
    """Network-wide (pooled over all stations) quantile of station-hour totals on training data,
    restricted to service hours (06-23) so that closed hours do not dilute the distribution."""
    m = (panel["ts"] <= train_end) & panel["ts"].dt.hour.between(6, 23)
    if train_start is not None:
        m &= panel["ts"] >= train_start
    return float((panel.loc[m, "entries"] + panel.loc[m, "exits"]).quantile(q))


# --------------------------------------------------------------------------- features
def total_frame(panel: pd.DataFrame, h: int = 1, spatial: bool = True, weather="auto", events="auto",
                cache: str | None = None, train_end: str = TRAIN_END, od_start_ym: str | None = None
                ) -> pd.DataFrame:
    """Feature frame for y = total(t+h) (entries+exits) built through features.make_features.

    make_features works on one target column, so we pass a panel whose ``entries`` column is the
    total and whose ``exits`` column is the exits (-> ``other_lag`` = exits at t). Entries at t are
    added explicitly. Spatial features (41_spatial_features) are merged on (ts, station)."""
    if cache is not None:
        from features import load_events, load_weather
        ext = f"w{int(load_weather() is not None)}e{int(load_events() is not None)}"
        f = CACHE / f"{cache}_h{h}_sp{int(spatial)}_{ext}.parquet"
        if f.exists():
            fr = pd.read_parquet(f)
            fr["station"] = fr["station"].astype(panel["station"].dtype)
            return fr
    p2 = panel[["ts", "station"]].copy()
    p2["entries"] = panel["entries"] + panel["exits"]
    p2["exits"] = panel["exits"]
    fr = make_features(p2, target="entries", horizon=h, weather=weather, events=events, train_end=train_end)
    fr["entries_t"] = panel["entries"].to_numpy(dtype="float64")
    fr["exits_t"] = panel["exits"].to_numpy(dtype="float64")
    if spatial:
        sp = mod("41_spatial_features").spatial_features(panel, h, train_end=train_end, od_start_ym=od_start_ym)
        fr = fr.merge(sp, on=["ts", "station"], how="left")
    if cache is not None:
        fr.to_parquet(f, index=False)
    return fr


def target_frame(panel: pd.DataFrame, target: str, h: int = 1, spatial: bool = True,
                 weather="auto", events="auto") -> pd.DataFrame:
    fr = make_features(panel, target=target, horizon=h, weather=weather, events=events)
    if spatial:
        sp = mod("41_spatial_features").spatial_features(panel, h)
        fr = fr.merge(sp, on=["ts", "station"], how="left")
    return fr


def spatial_cols(fr: pd.DataFrame) -> list[str]:
    return [c for c in fr.columns if c.startswith(("nb", "line_", "net_", "od_"))]


def base_cols(fr: pd.DataFrame) -> list[str]:
    sp = set(spatial_cols(fr))
    return [c for c in feature_columns(fr) if c not in sp and c not in ("crowd_thr", "label", "label_abs",
                                                                         "label_dual")]


def split(fr: pd.DataFrame, train_end=TRAIN_END, valid=(VALID_START, VALID_END), test=(TEST_START, TEST_END),
          train_start: str | None = None):
    fr = fr.dropna(subset=["y", "lag_168"])
    t = fr["ts_target"]
    mtr = t <= train_end
    if train_start is not None:
        mtr &= t >= train_start
    return (fr[mtr], fr[(t >= valid[0]) & (t <= valid[1])], fr[(t >= test[0]) & (t <= test[1])])


# --------------------------------------------------------------------------- models
def fit_scaled_reg(tr, va, cols, rounds=3000):
    """LightGBM-scaled (as in 04_baselines_t1): target y/scale, weight=scale, L2 loss."""
    dtr = lgb.Dataset(tr[cols], tr.y / tr.scale, weight=tr.scale, categorical_feature=["station"])
    dva = lgb.Dataset(va[cols], va.y / va.scale, weight=va.scale, reference=dtr)
    return lgb.train({**LGB_REG, "objective": "regression"}, dtr, num_boost_round=rounds, valid_sets=[dva],
                     callbacks=[lgb.early_stopping(100, verbose=False)])


def pred_scaled(m, fr, cols):
    return np.clip(m.predict(fr[cols], num_iteration=m.best_iteration) * fr.scale.to_numpy(), 0, None)


def _focal_obj(gamma: float = 2.0, alpha: float = 0.75):
    """Binary focal loss (Lin et al. 2017) gradient/hessian for LightGBM (numerical 2nd derivative)."""
    def loss(z, y):
        p = 1 / (1 + np.exp(-z))
        pt = np.where(y == 1, p, 1 - p)
        at = np.where(y == 1, alpha, 1 - alpha)
        return -at * (1 - pt) ** gamma * np.log(np.clip(pt, 1e-12, 1))

    def obj(preds, data):
        y = data.get_label()
        eps = 1e-4
        l0, lp, lm = loss(preds, y), loss(preds + eps, y), loss(preds - eps, y)
        g = (lp - lm) / (2 * eps)
        hss = np.maximum((lp - 2 * l0 + lm) / eps ** 2, 1e-6)
        return g, hss
    return obj


def fit_clf(tr, va, cols, label="label", mode="none", rounds=2000, seed=0):
    """Binary LightGBM classifier with an imbalance strategy:
    none | class_weight (pos weight = neg/pos) | focal (focal loss) | undersample (neg:pos = 5:1)."""
    ytr, yva = tr[label].to_numpy(), va[label].to_numpy()
    Xtr = tr[cols]
    params = {**LGB_CLF, "objective": "binary", "metric": "average_precision", "seed": seed}
    w = None
    if mode == "class_weight":
        w = np.where(ytr == 1, (ytr == 0).sum() / max(ytr.sum(), 1), 1.0)
    elif mode == "undersample":
        rng = np.random.default_rng(seed)
        neg = np.flatnonzero(ytr == 0)
        keep = np.concatenate([np.flatnonzero(ytr == 1),
                               rng.choice(neg, size=min(len(neg), 5 * int(ytr.sum())), replace=False)])
        keep.sort()
        Xtr, ytr = Xtr.iloc[keep], ytr[keep]
    elif mode == "focal":
        params["objective"] = _focal_obj()
    dtr = lgb.Dataset(Xtr, ytr, weight=w, categorical_feature=["station"])
    dva = lgb.Dataset(va[cols], yva, reference=dtr)
    m = lgb.train(params, dtr, num_boost_round=rounds, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    return m


def pred_clf(m, fr, cols):
    z = m.predict(fr[cols], num_iteration=m.best_iteration, raw_score=True)
    return 1 / (1 + np.exp(-z))


def to_md(df: pd.DataFrame) -> str:
    """Minimal markdown table (avoids the optional `tabulate` dependency)."""
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for r in df.itertuples(index=False):
        lines.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in r) + " |")
    return "\n".join(lines)
