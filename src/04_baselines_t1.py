"""Reproduce progress-report Table 1: baselines vs LightGBM vs LightGBM-scaled for t+h.

Usage: python src/04_baselines_t1.py [--horizon 1]
Writes results/table1_regression_h{h}.csv and results/pred_test_h{h}_{target}.parquet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import RESULTS, TEST_END, TEST_START, TRAIN_END, VALID_END, VALID_START  # noqa: E402
from features import feature_columns, load_panel, make_features  # noqa: E402
from metrics import reg_metrics  # noqa: E402

LGB_PARAMS = dict(learning_rate=0.05, num_leaves=127, min_data_in_leaf=100, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=0)


def split(fr: pd.DataFrame):
    t = fr["ts_target"]
    fr = fr.dropna(subset=["y", "lag_168"])
    t = fr["ts_target"]
    return (fr[t <= TRAIN_END], fr[(t >= VALID_START) & (t <= VALID_END)], fr[(t >= TEST_START) & (t <= TEST_END)])


def fit_lgb(tr, va, cols, objective, scaled: bool):
    if scaled:
        ytr, yva = tr.y / tr.scale, va.y / va.scale
        wtr, wva = tr.scale, va.scale
    else:
        ytr, yva, wtr, wva = tr.y, va.y, None, None
    dtr = lgb.Dataset(tr[cols], ytr, weight=wtr, categorical_feature=["station"])
    dva = lgb.Dataset(va[cols], yva, weight=wva, reference=dtr)
    m = lgb.train({**LGB_PARAMS, "objective": objective}, dtr, num_boost_round=3000, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    return m


def predict(m, fr, cols, scaled):
    p = m.predict(fr[cols], num_iteration=m.best_iteration)
    return np.clip(p * fr.scale.to_numpy() if scaled else p, 0, None)


def run(target: str, h: int, panel: pd.DataFrame):
    fr = make_features(panel, target=target, horizon=h)
    tr, va, te = split(fr)
    rows, preds = [], te[["ts_target", "station", "y", "hour", "is_holiday"]].copy()
    if "precip" in te:
        preds["precip"] = te["precip"]
    base = {"Last Value": te[f"lag_{h}"], "Seasonal Naive": te["lag_168"], "Historical Average": te["hist_avg"]}
    for name, p in base.items():
        preds[name] = p.to_numpy()
    # first version: original features, Poisson loss
    cols_v1 = [c for c in feature_columns(fr) if c not in ("hist_avg", "lag_336", "lag_48", "diff_1")
               and not c.startswith("other_lag") and c not in ("is_makeup_workday", "is_long_weekend",
               "holiday_block_len", "days_to_next_holiday", "days_since_last_holiday", "month")]
    m1 = fit_lgb(tr, va, cols_v1, "poisson", scaled=False)
    preds["LightGBM"] = predict(m1, te, cols_v1, False)
    cols_v2 = feature_columns(fr)
    m2 = fit_lgb(tr, va, cols_v2, "regression", scaled=True)
    preds["LightGBM (scaled)"] = predict(m2, te, cols_v2, True)
    for name in list(base) + ["LightGBM", "LightGBM (scaled)"]:
        rows.append({"target": target, "horizon": h, "method": name, **reg_metrics(te.y, preds[name], te)})
    imp = pd.Series(m1.feature_importance("gain"), index=cols_v1)
    imp = (imp / imp.sum()).sort_values(ascending=False).round(4)
    imp.to_csv(RESULTS / f"importance_v1_h{h}_{target}.csv")
    imp2 = pd.Series(m2.feature_importance("gain"), index=cols_v2)
    (imp2 / imp2.sum()).sort_values(ascending=False).round(4).to_csv(RESULTS / f"importance_scaled_h{h}_{target}.csv")
    preds.to_parquet(RESULTS / f"pred_test_h{h}_{target}.parquet", index=False)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=1)
    a = ap.parse_args()
    panel = load_panel()
    rows = []
    for target in ("entries", "exits"):
        rows += run(target, a.horizon, panel)
        print(pd.DataFrame(rows).to_string(index=False), flush=True)
    pd.DataFrame(rows).to_csv(RESULTS / f"table1_regression_h{a.horizon}.csv", index=False)


if __name__ == "__main__":
    main()
