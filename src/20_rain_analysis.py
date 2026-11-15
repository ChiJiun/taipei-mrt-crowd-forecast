"""Workstream B task 5: why is the rainy-hour error so high, and can rain-derived features help?

Breaks down errors of LightGBM-scaled (report version) and LightGBM-B on rainy vs dry target hours by
rain intensity, day type, hour, rain onset (first rainy hour) vs continuing rain, station, and sign (bias),
then runs a feature ablation: LightGBM-B with vs without the rain-derivative features
(precip change, previous-1/3h rain, 6h accumulation, rain onset, consecutive rainy hours, rain x holiday,
rain at origin).  Evaluated on validation + test (2026-08..09) to have more rainy hours, plus test only.

Needs data/external/weather_hourly.csv. Run after 20_regression.py (uses its saved predictions) or standalone.
Usage: python src/20_rain_analysis.py [--quick] [--target entries] [--horizon 1]
Writes results/reg_rain_tables.md (auto tables; the narrative is results/reg_rain_analysis.md), results/reg_rain_breakdown.csv, figures/reg_rain_error.png
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import FIGURES, PEAK_HOURS, RAIN_MM, RESULTS, TRAIN_END, VALID_START  # noqa: E402
from features import load_panel  # noqa: E402
from plotting import plt  # noqa: E402

R = importlib.import_module("20_reglib")
RAIN_DERIV = ["precip_prev1", "precip_prev3", "precip_6h", "precip_change", "rain_onset", "rain_run",
              "rain_x_holiday", "precip_origin"]
DT_NAME = {0: "平日", 1: "週六", 2: "週日／假日"}


def hour_band(hh, dt):
    if dt == 0 and hh in PEAK_HOURS:
        return "平日尖峰"
    if 6 <= hh <= 9 or 16 <= hh <= 19:
        return "尖峰前後／假日同時段"
    if 10 <= hh <= 15:
        return "白天離峰"
    if 20 <= hh <= 23:
        return "晚間"
    return "深夜清晨"


def summarize(g, methods):
    r = {"n": len(g), "mean_y": g.y.mean()}
    for m in methods:
        e = g[m] - g.y
        r[f"MAE_{m}"] = np.abs(e).mean()
        r[f"bias_{m}"] = e.mean()
        r[f"relMAE%_{m}"] = 100 * np.abs(e).sum() / max(g.y.sum(), 1)
    return pd.Series(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--target", default="entries")
    ap.add_argument("--horizon", type=int, default=1)
    a = ap.parse_args()
    if not R.weather_available():
        sys.exit("weather_hourly.csv not found: rain analysis needs workstream A's weather data")
    tgt, h = a.target, a.horizon
    params = {"learning_rate": 0.1}
    panel = load_panel()
    fr = R.build(panel, tgt, h)
    # "recent level correction" (suggested by workstream C's SARIMAX finding): deviation of the last
    # 2-3 origin hours from the station profile, in scale units
    g = fr.groupby("station", observed=True)["anom_origin_s"]
    fr["anom_prev1_s"] = g.shift(1)
    fr["anom_mean3_s"] = g.transform(lambda x: x.rolling(3, min_periods=1).mean())
    fr["anom_trend_s"] = fr["anom_origin_s"] - fr["anom_prev1_s"]
    recent = ["anom_prev1_s", "anom_mean3_s", "anom_trend_s"]
    tr, va, te = R.split(fr)
    full = [c for c in R.cols_full(fr) if c not in recent]
    # Evaluate on valid+test: models trained on <= TRAIN_END with early stopping on a slice of training
    # (last 4 weeks) so August is also out-of-sample.
    es_start = pd.Timestamp(TRAIN_END) - pd.Timedelta(days=28)
    tr2 = tr[tr.ts_target < es_start]
    es = tr[tr.ts_target >= es_start]
    ev = pd.concat([va, te])
    out = ev[["ts_target", "station", "y", "hour", "daytype", "is_holiday", "precip", "precip_prev1", "rain_onset",
              "rain_run", "scale", "hist_dt"]].copy()
    variants = {
        "LightGBM-scaled": (R.cols_scaled(fr), "regression"),
        "B 無降雨衍生特徵": ([c for c in full if c not in RAIN_DERIV], R.B_OBJ),
        "B 無任何天氣": ([c for c in full if c not in RAIN_DERIV and c not in ("precip", "precip_3h", "temp", "rh")
                     and c not in (R.load_weather("obs").columns)], R.B_OBJ),
        "LightGBM-B": (full, R.B_OBJ),
        "B + 近期水準修正": (full + recent, R.B_OBJ),
    }
    for name, (cols, obj) in variants.items():
        m = R.fit(tr2, es, cols, obj, True, params=params)
        out[name] = R.predict(m, ev, cols, True)
        print(f"{name}: iter={m.best_iteration}", flush=True)
    methods = list(variants)
    out["rain"] = out.precip >= RAIN_MM
    out["period"] = np.where(out.ts_target >= VALID_START, "valid+test", "")
    out["is_test"] = out.ts_target >= pd.Timestamp("2026-09-01")
    out["intensity"] = pd.cut(out.precip, [-1, 0.0999, 1, 3, 10, 1e9],
                              labels=["0", "0.1–1", "1–3", "3–10", "≥10"])
    out["dtn"] = out.daytype.map(DT_NAME)
    out["band"] = [hour_band(a_, b_) for a_, b_ in zip(out.hour, out.daytype)]
    out["phase"] = np.select([~out.rain, out.rain_onset == 1, out.rain_run <= 3],
                             ["無雨", "降雨第 1 小時", "第 2–3 小時"], "第 4 小時以上")
    sections = []

    def tab(by, data=out, title=""):
        t = data.groupby(by, observed=True).apply(summarize, methods=methods, include_groups=False)
        t = t.reset_index()
        sections.append((title, t))
        return t

    tab("rain", title="雨天 vs 無雨（2026-08~09）")
    tab("rain", out[out.is_test], title="雨天 vs 無雨（僅測試集 2026-09）")
    tab("intensity", title="依降雨強度（mm/h）")
    tab(["rain", "dtn"], title="依日別")
    tab(["rain", "band"], title="依時段")
    tab("phase", title="依降雨階段（雨開始第幾小時）")
    # stations with largest rain excess error (relative)
    st = out.groupby(["station", "rain"], observed=True).apply(summarize, methods=["LightGBM-B"],
                                                               include_groups=False).unstack("rain")
    st.columns = [f"{a_}_{'rain' if b_ else 'dry'}" for a_, b_ in st.columns]
    st["rain_dry_ratio"] = st["MAE_LightGBM-B_rain"] / st["MAE_LightGBM-B_dry"]
    st["extra_MAE"] = st["MAE_LightGBM-B_rain"] - st["MAE_LightGBM-B_dry"]
    st = st.sort_values("extra_MAE", ascending=False).reset_index()
    sections.append(("雨天額外誤差最大的 15 站（LightGBM-B）", st.head(15)[
        ["station", "n_rain", "mean_y_rain", "mean_y_dry", "MAE_LightGBM-B_rain", "MAE_LightGBM-B_dry",
         "rain_dry_ratio", "bias_LightGBM-B_rain"]]))
    # mixing effect: rain hours are concentrated in which hours? share of rainy hours that are daytime/peak
    mix = out.groupby("rain").agg(mean_y=("y", "mean"), share_peak=("band", lambda s: (s == "平日尖峰").mean()),
                                  share_daytime=("hour", lambda s: s.between(7, 20).mean())).reset_index()
    sections.append(("組成效應：雨天小時的時段分布", mix))
    # "same mix" comparison: dry-hour MAE reweighted to the rainy hours' (station, hour, daytype) mix
    key = ["station", "hour", "daytype"]
    dry = out[~out.rain].groupby(key, observed=True)["LightGBM-B"].apply(lambda s: s.name)  # placeholder
    del dry
    out["ae_B"] = (out["LightGBM-B"] - out.y).abs()
    cell = out[~out.rain].groupby(key, observed=True)["ae_B"].mean().rename("dry_cell_MAE")
    rr = out[out.rain].join(cell, on=key)
    mix_adj = pd.DataFrame({"指標": ["雨天 MAE", "無雨 MAE（全部）", "無雨 MAE（以雨天的站×時×日別組成加權）"],
                            "LightGBM-B": [rr.ae_B.mean(), out[~out.rain].ae_B.mean(), rr.dry_cell_MAE.mean()]})
    sections.append(("控制組成後的雨天誤差", mix_adj))

    allb = pd.concat([t.assign(section=s) for s, t in sections], ignore_index=True)
    allb.round(2).to_csv(RESULTS / "reg_rain_breakdown.csv", index=False)

    # figure: MAE by intensity & bias by phase
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    t1 = sections[2][1]
    x = np.arange(len(t1))
    for i, mt in enumerate(methods):
        axes[0].bar(x + (i - 2) * 0.16, t1[f"relMAE%_{mt}"], 0.16, label=mt)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(t1["intensity"].astype(str))
    axes[0].set_xlabel("降雨強度 (mm/h)")
    axes[0].set_ylabel("相對 MAE（%，ΣAE/Σy）")
    axes[0].legend(fontsize=7)
    t2 = sections[5][1]
    x = np.arange(len(t2))
    for i, mt in enumerate(methods):
        axes[1].bar(x + (i - 2) * 0.16, t2[f"bias_{mt}"], 0.16, label=mt)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(t2["phase"], fontsize=8)
    axes[1].axhline(0, color="gray", lw=.6)
    axes[1].set_ylabel("平均偏差（預測 − 實際）")
    axes[1].set_title("降雨階段的偏差")
    t3 = sections[3][1]
    lab = [f"{'雨' if r else '晴'}·{d}" for r, d in zip(t3.rain, t3.dtn)]
    x = np.arange(len(t3))
    for i, mt in enumerate(methods):
        axes[2].bar(x + (i - 2) * 0.16, t3[f"bias_{mt}"], 0.16, label=mt)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(lab, fontsize=8, rotation=30)
    axes[2].axhline(0, color="gray", lw=.6)
    axes[2].set_title("雨天 × 日別的偏差")
    axes[0].set_title("各降雨強度的相對誤差")
    fig.tight_layout()
    fig.savefig(FIGURES / "reg_rain_error.png", dpi=150)
    plt.close(fig)

    md = [f"# 雨天誤差分析（{'進' if tgt == 'entries' else '出'}站 t+{h}）\n",
          "評估期：2026-08（驗證）+ 2026-09（測試），模型以 ~2026-07 訓練、最後 4 週作 early stopping，"
          "因此 8 月也是樣本外。雨天 = 目標小時降雨 ≥ 1 mm。relMAE% = ΣAE/Σy；bias = 平均（預測 − 實際），"
          "負值代表低估。\n", "<!-- AUTO TABLES -->\n"]
    for s, t in sections:
        md += [f"## {s}\n", R.md_table(t.round(2), "{:.2f}"), ""]
    (RESULTS / "reg_rain_tables.md").write_text("\n".join(md), encoding="utf-8")
    out.drop(columns=["period"]).to_parquet(RESULTS / f"pred_rain_h{h}_{tgt}.parquet", index=False)
    print("\n".join(md))


if __name__ == "__main__":
    main()
