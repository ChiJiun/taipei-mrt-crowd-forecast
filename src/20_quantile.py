"""Workstream B task 3: quantile prediction intervals (LightGBM quantile P10 / P50 / P90).

Quantile models are fit on the scaled target y/scale with weight = scale, so the weighted pinball loss equals
the pinball loss on the original scale. The raw P10-P90 interval is then calibrated on the validation month
(split-conformal CQR: widen/shrink both ends by the (1-a) quantile of the conformity score), and both raw and
calibrated intervals are evaluated on the test month: coverage, width, pinball loss — overall, weekday peak,
off-peak, and (with weather) rainy hours.

Usage: python src/20_quantile.py [--quick]   (quick: entries h=1 only, learning_rate 0.1)
Writes results/reg_quantile.csv, results/reg_quantile.md, figures/reg_interval_taipei_main.png
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import FIGURES, RAIN_MM, RESULTS  # noqa: E402
from features import load_panel  # noqa: E402
from metrics import peak_mask  # noqa: E402
from plotting import plt  # noqa: E402

R = importlib.import_module("20_reglib")
QS = (0.1, 0.5, 0.9)
NOMINAL = QS[2] - QS[0]
PLOT_STATION, PLOT_START, PLOT_END = "台北車站", "2026-09-14 00:00", "2026-09-20 23:00"


def fit_quantiles(tr, va, cols, params):
    models = {}
    for a in QS:
        models[a] = R.fit(tr, va, cols, "quantile", True, params=params, alpha=a, rounds=2000, es=50)
        print(f"    q={a}: iter={models[a].best_iteration}", flush=True)
    return models


def predict_q(models, fr, cols):
    P = np.column_stack([R.predict(models[a], fr, cols, True) for a in QS])
    P.sort(axis=1)  # avoid quantile crossing
    return P


def cqr_offset(y, lo, hi, cov=NOMINAL):
    """Split-conformal offset, computed in scaled units per station-size (score / scale)."""
    s = np.maximum(lo - y, y - hi)
    n = len(s)
    return float(np.quantile(s, min(1.0, np.ceil((n + 1) * cov) / n)))


def evaluate(y, P, lo, hi, frame, label):
    masks = {"all": np.ones(len(y), bool), "peak": peak_mask(frame), "offpeak": ~peak_mask(frame)}
    if "precip" in frame:
        masks["rain"] = (frame["precip"] >= RAIN_MM).to_numpy()
    rows = []
    for k, m in masks.items():
        if not m.any():
            continue
        yy = y[m]
        r = {"interval": label, "subset": k, "n": int(m.sum()),
             "coverage": round(float(((yy >= lo[m]) & (yy <= hi[m])).mean()), 3),
             "below_P10": round(float((yy < lo[m]).mean()), 3), "above_P90": round(float((yy > hi[m]).mean()), 3),
             "mean_width": round(float((hi[m] - lo[m]).mean()), 1),
             "MAE_P50": round(float(np.abs(yy - P[m, 1]).mean()), 2)}
        for j, a in enumerate(QS):
            r[f"pinball_P{int(a * 100)}"] = round(R.pinball(yy, P[m, j], a), 2)
        # interval score (Gneiting & Raftery): width + 2/alpha * miss distance
        al = 1 - NOMINAL
        r["interval_score"] = round(float(np.mean((hi[m] - lo[m]) + 2 / al * (lo[m] - yy) * (yy < lo[m])
                                                  + 2 / al * (yy - hi[m]) * (yy > hi[m]))), 1)
        rows.append(r)
    return rows


def plot_station(te, P, lo_c, hi_c, target, h):
    m = ((te["station"] == PLOT_STATION) & (te["ts_target"] >= PLOT_START) & (te["ts_target"] <= PLOT_END)).to_numpy()
    if not m.any():
        return
    t = te["ts_target"].to_numpy()[m]
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.fill_between(t, lo_c[m], hi_c[m], color="#9ecae1", alpha=0.6, label="P10–P90（校準後）")
    ax.plot(t, P[m, 1], color="#08519c", lw=1.2, label="P50 預測")
    ax.plot(t, te["y"].to_numpy()[m], color="black", lw=0.9, ls="--", label="實際")
    y = te["y"].to_numpy()[m]
    out = (y < lo_c[m]) | (y > hi_c[m])
    ax.scatter(t[out], y[out], s=12, color="#d62728", zorder=5, label="落在區間外")
    if "precip" in te:
        ax2 = ax.twinx()
        ax2.bar(t, te["precip"].to_numpy()[m], width=1 / 24, color="#31a354", alpha=0.35)
        ax2.set_ylabel("降雨 (mm)", color="#31a354")
        ax2.set_ylim(0, max(10, np.nanmax(te["precip"].to_numpy()[m]) * 3))
    cov = 1 - out.mean()
    ax.set_title(f"台北車站{'進' if target == 'entries' else '出'}站量 t+{h} 預測區間（2026/9/14–9/20，覆蓋率 {cov:.0%}）")
    ax.set_ylabel("人次 / 小時")
    ax.legend(loc="upper left", ncol=4, fontsize=8)
    fig.autofmt_xdate()
    fig.savefig(FIGURES / ("reg_interval_taipei_main.png" if (target, h) == ("entries", 1)
                           else f"reg_interval_taipei_main_h{h}_{target}.png"), dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    params = {"learning_rate": 0.1}
    panel = load_panel()
    combos = [("entries", 1)] if a.quick else [("entries", 1), ("exits", 1), ("entries", 2), ("exits", 2)]
    rows = []
    for tgt, h in combos:
        print(f"{tgt} h={h}", flush=True)
        fr = R.build(panel, tgt, h)
        tr, va, te = R.split(fr)
        cols = R.cols_full(fr)
        models = fit_quantiles(tr, va, cols, params)
        Pv, Pt = predict_q(models, va, cols), predict_q(models, te, cols)
        sv, st = va["scale"].to_numpy(), te["scale"].to_numpy()
        # CQR in scaled units so the correction is proportional to station size
        off = cqr_offset(va.y.to_numpy() / sv, Pv[:, 0] / sv, Pv[:, 2] / sv)
        lo_c, hi_c = np.clip(Pt[:, 0] - off * st, 0, None), Pt[:, 2] + off * st
        y = te.y.to_numpy()
        for r in evaluate(y, Pt, Pt[:, 0], Pt[:, 2], te, "raw") + evaluate(y, Pt, lo_c, hi_c, te, "CQR"):
            rows.append({"target": tgt, "horizon": h, "cqr_offset_scaled": round(off, 4), **r})
        plot_station(te, Pt, lo_c, hi_c, tgt, h)
        print(pd.DataFrame(rows).tail(8).to_string(index=False), flush=True)
    df = pd.DataFrame(rows)
    suffix = "_quick" if a.quick else ""
    df.to_csv(RESULTS / f"reg_quantile{suffix}.csv", index=False)
    md = ["# 分位數預測區間（LightGBM quantile P10/P50/P90，測試集 2026-09）\n",
          f"名目覆蓋率 {NOMINAL:.0%}。raw = 直接用 P10/P90；CQR = 以驗證集（2026-08）做 conformal 校準。"
          "interval_score 越低越好（寬度 + 未覆蓋懲罰）。\n"]
    show = ["target", "horizon", "interval", "subset", "coverage", "below_P10", "above_P90", "mean_width",
            "MAE_P50", "pinball_P10", "pinball_P50", "pinball_P90", "interval_score"]
    md.append(R.md_table(df[show], "{:.3g}"))
    (RESULTS / f"reg_quantile{suffix}.md").write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()
