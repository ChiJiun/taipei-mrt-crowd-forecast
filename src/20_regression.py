"""Workstream B task 1: formal t+1 / t+2 regression experiments (entries / exits).

Methods: Last Value, Seasonal Naive, Historical Average, LightGBM (report v1, Poisson), LightGBM-scaled
(report version), LightGBM-B (improved: normalised lags, day-type profiles, origin anomaly, weather
derivatives, L2 loss on the scaled target) and, when weather exists, LightGBM-B without weather (ablation).

Usage:
  python src/20_regression.py            # full run: entries/exits x h in {1,2}
  python src/20_regression.py --quick    # entries, h=1 only, faster learning rate
Writes results/reg_main.csv, results/reg_main.md, results/reg_ablation.csv,
      results/pred_reg_h{h}_{target}.parquet (gitignored) and data/interim/reg_models/*.txt (gitignored).
"""
from __future__ import annotations

import argparse
import importlib
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import INTERIM, RESULTS  # noqa: E402
from features import load_panel  # noqa: E402
from metrics import reg_metrics  # noqa: E402

R = importlib.import_module("20_reglib")
MODEL_DIR = INTERIM / "reg_models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

WEATHER_COLS = ("temp", "precip", "rh")


def weather_cols(fr):
    """All weather-derived columns present in the frame (shared + B)."""
    base = [c for c in fr.columns if c in WEATHER_COLS or c.startswith("precip") or c in (
        "rain_onset", "rain_run", "rain_x_holiday")]
    # any extra numeric columns that came from the weather file
    w = R.load_weather("obs")
    if w is not None:
        base += [c for c in w.columns if c != "ts" and c in fr.columns]
    return sorted(set(base))


def run(panel, target, h, quick=False, ablation=True):
    params = {"learning_rate": 0.1} if quick else {}
    fr = R.build(panel, target, h)
    tr, va, te = R.split(fr)
    keep = ["ts", "ts_target", "station", "y", "hour", "dow", "is_holiday", "daytype", "scale"] + \
        [c for c in ("precip", "precip_prev1", "rain_onset", "rain_run", "event_flag") if c in te]
    preds = te[keep].copy()
    preds["Last Value"] = te[f"lag_{h}"].to_numpy()
    preds["Seasonal Naive"] = te["lag_168"].to_numpy()
    preds["Historical Average"] = te["hist_avg"].to_numpy()
    info = {}

    def go(name, cols, objective, scaled, save=None):
        t0 = time.time()
        prm = {**R.B_PARAMS, **params} if name.startswith("LightGBM-B") else params
        m = R.fit(tr, va, cols, objective, scaled, params=prm)
        preds[name] = R.predict(m, te, cols, scaled)
        info[name] = dict(best_iter=m.best_iteration, n_feat=len(cols), sec=round(time.time() - t0),
                          valid_MAE=round(float(np.abs(R.predict(m, va, cols, scaled) - va.y).mean()), 2))
        print(f"  {target} h={h} {name}: iter={m.best_iteration} {info[name]}", flush=True)
        if save:
            m.save_model(str(MODEL_DIR / f"{save}_h{h}_{target}.txt"), num_iteration=m.best_iteration)
            pd.Series(cols).to_csv(MODEL_DIR / f"{save}_h{h}_{target}_cols.csv", index=False)
        return m

    go("LightGBM", R.cols_v1(fr), "poisson", False)
    go("LightGBM-scaled", R.cols_scaled(fr), "regression", True)
    full = R.cols_full(fr)
    go("LightGBM-B", full, R.B_OBJ, True, save="lgbB")
    abl = []
    if ablation:
        wc = weather_cols(fr)
        if wc:
            go("LightGBM-B (no weather)", [c for c in full if c not in wc], R.B_OBJ, True)
        if (target, h) == ("entries", 1):  # L1 is ~8x slower; run once for the record
            go("LightGBM-B (L1 loss)", full, "regression_l1", True)
        ev = [c for c in full if c.startswith("event")]
        if ev:
            go("LightGBM-B (no events)", [c for c in full if c not in ev], R.B_OBJ, True)
    rows = []
    for name in [c for c in preds.columns if c not in keep]:
        r = {"target": target, "horizon": h, "method": name, **reg_metrics(te.y, preds[name], te)}
        r.update(info.get(name, {}))
        rows.append(r)
    preds.to_parquet(RESULTS / f"pred_reg_h{h}_{target}.parquet", index=False)
    return rows


def to_md(df: pd.DataFrame) -> str:
    out = []
    for (tgt, h), g in df.groupby(["target", "horizon"], sort=False):
        cols = [c for c in ("method", "MAE", "RMSE", "peak_MAE", "rain_MAE", "dry_MAE", "MAE_vs_HA_%") if c in g]
        out.append(f"### {'進站' if tgt == 'entries' else '出站'} t+{h}（測試集 2026-09）\n")
        out.append(R.md_table(g[cols]))
        out.append("")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="entries h=1 only, learning_rate 0.1")
    ap.add_argument("--targets", default="entries,exits")
    ap.add_argument("--horizons", default="1,2")
    ap.add_argument("--no-ablation", action="store_true")
    a = ap.parse_args()
    targets = ["entries"] if a.quick else a.targets.split(",")
    hs = [1] if a.quick else [int(x) for x in a.horizons.split(",")]
    print(f"weather: {R.weather_available()}  events: {R.load_events() is not None}", flush=True)
    panel = load_panel()
    rows = []
    for h in hs:
        for tgt in targets:
            rows += run(panel, tgt, h, quick=a.quick, ablation=not a.no_ablation)
    df = pd.DataFrame(rows)
    ha = df[df.method == "Historical Average"].set_index(["target", "horizon"])["MAE"]
    df["MAE_vs_HA_%"] = [round(100 * (1 - r.MAE / ha[(r.target, r.horizon)]), 1) for r in df.itertuples()]
    main_methods = ["Last Value", "Seasonal Naive", "Historical Average", "LightGBM", "LightGBM-scaled", "LightGBM-B"]
    suffix = "_quick" if a.quick else ""
    df[df.method.isin(main_methods)].drop(columns=["best_iter", "n_feat", "sec", "valid_MAE"], errors="ignore") \
        .to_csv(RESULTS / f"reg_main{suffix}.csv", index=False)
    df.to_csv(RESULTS / f"reg_ablation{suffix}.csv", index=False)
    hdr = (f"# 迴歸主結果（工作流 B）\n\n天氣特徵：{'有（觀測值）' if R.weather_available() else '無'}；"
           f"活動特徵：{'有' if R.load_events() is not None else '無'}。MAE_vs_HA_% = 相對 Historical Average 的 MAE 改善。"
           "peak_MAE = 平日 7、8、17、18 時；rain/dry = 目標小時降雨 ≥ 1 mm / < 1 mm。\n\n")
    (RESULTS / f"reg_main{suffix}.md").write_text(hdr + to_md(df[df.method.isin(main_methods)]) +
                                                  "\n\n## 消融（含改良模型變體）\n\n" + to_md(df), encoding="utf-8")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
