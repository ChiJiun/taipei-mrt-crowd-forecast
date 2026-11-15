"""Workstream B task 4: SHAP analysis of the improved model (LightGBM-B, entries, t+1).

The model predicts y/scale, so SHAP values are in units of "station weekly mean level"; multiplying by the
row's scale turns them into additive contributions in persons/hour (pred = scale * (base + sum shap)).
Feature groups: time-of-day / calendar, holiday, lags (recent / seasonal), profile, weather, events, station.

Usage: python src/20_shap.py [--quick] [--target entries] [--horizon 1] [--n 20000]
  Uses the model saved by 20_regression.py if present (data/interim/reg_models), otherwise retrains.
Writes results/reg_shap_importance.csv, results/reg_shap_groups.csv, figures/reg_shap_summary.png,
       figures/reg_shap_group_bar.png, figures/reg_shap_dependence_precip_holiday.png (if weather),
       figures/reg_shap_dependence_hour.png, figures/reg_shap_holiday_by_hour.png
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap

sys.path.insert(0, str(Path(__file__).parent))
from config import FIGURES, INTERIM, RESULTS  # noqa: E402
from features import load_panel  # noqa: E402
from plotting import plt  # noqa: E402

R = importlib.import_module("20_reglib")

GROUPS = [
    ("天氣", lambda c: c in ("temp", "rh", "rain_onset", "rain_run", "rain_x_holiday") or c.startswith("precip")
     or c in ("wind", "wind_speed", "cloud", "apparent_temp")),
    ("活動", lambda c: c.startswith("event")),
    ("假日／日別", lambda c: c in ("is_holiday", "is_makeup_workday", "is_long_weekend", "holiday_block_len",
                               "days_to_next_holiday", "days_since_last_holiday", "daytype", "dow")),
    ("時段", lambda c: c in ("hour", "month")),
    ("近期延遲（t, t-1, t-2）", lambda c: c.startswith("lag_") and c.split("_")[1].isdigit()
     and int(c.split("_")[1]) <= 4 or c in ("diff_1", "anom_origin_s", "ratio_origin", "roll_24", "roll_24_s")
     or c.startswith("other_lag")),
    ("週期延遲（24/48/168/336h）", lambda c: c.startswith("lag_") or c in ("ratio_168", "roll_168")),
    ("歷史平均剖面", lambda c: c.startswith("hist_")),
    ("站點／量級", lambda c: c in ("station", "log_scale")),
]


def group_of(c):
    for g, f in GROUPS:
        if f(c):
            return g
    return "其他"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--target", default="entries")
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--n", type=int, default=8000)
    a = ap.parse_args()
    n = 2000 if a.quick else a.n
    tgt, h = a.target, a.horizon
    panel = load_panel()
    fr = R.build(panel, tgt, h)
    tr, va, te = R.split(fr)
    mf = INTERIM / "reg_models" / f"lgbB_h{h}_{tgt}.txt"
    cols = R.cols_full(fr)
    if mf.exists() and list(pd.read_csv(mf.with_name(mf.stem + "_cols.csv")).iloc[:, 0]) == cols:
        m = lgb.Booster(model_file=str(mf))
        print("loaded", mf)
    else:
        print("training LightGBM-B ...", flush=True)
        m = R.fit(tr, va, cols, R.B_OBJ, True, params={"learning_rate": 0.1})
    rng = np.random.default_rng(0)
    # oversample rainy / holiday rows so their effects are visible, but importance uses the plain sample
    idx = rng.choice(len(te), size=min(n, len(te)), replace=False)
    X = te.iloc[idx][cols]
    sc = te.iloc[idx]["scale"].to_numpy()
    expl = shap.TreeExplainer(m)
    sv = expl.shap_values(X)
    svp = sv * sc[:, None]  # persons / hour
    imp = pd.DataFrame({"feature": cols, "mean_abs_shap_scaled": np.abs(sv).mean(0),
                        "mean_abs_shap_persons": np.abs(svp).mean(0)})
    imp["group"] = imp.feature.map(group_of)
    imp["share_%"] = (100 * imp.mean_abs_shap_persons / imp.mean_abs_shap_persons.sum()).round(2)
    imp = imp.sort_values("mean_abs_shap_persons", ascending=False).round(4)
    imp.to_csv(RESULTS / "reg_shap_importance.csv", index=False)
    grp = imp.groupby("group")[["mean_abs_shap_persons", "share_%"]].sum().sort_values("share_%", ascending=False)
    # group-level |sum of SHAP| (interaction-free total contribution of a group per row)
    gs = {g: np.abs(svp[:, [cols.index(c) for c in imp.feature[imp.group == g]]].sum(1)).mean()
          for g in grp.index}
    grp["mean_abs_group_shap_persons"] = pd.Series(gs).round(2)
    grp.round(2).to_csv(RESULTS / "reg_shap_groups.csv")
    print(imp.head(25).to_string(index=False))
    print(grp)

    # summary plot (scaled units)
    plt.figure()
    shap.summary_plot(sv, X.assign(station=X.station.cat.codes), max_display=20, show=False)
    plt.title(f"SHAP summary（LightGBM-B，{'進' if tgt == 'entries' else '出'}站 t+{h}，單位：站點週平均量級）")
    plt.savefig(FIGURES / "reg_shap_summary.png", dpi=150)
    plt.close("all")

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.barh(grp.index[::-1], grp["mean_abs_group_shap_persons"][::-1], color="#3182bd")
    ax.set_xlabel("平均 |SHAP|（人次／小時，群組加總）")
    ax.set_title("特徵群組影響力")
    fig.savefig(FIGURES / "reg_shap_group_bar.png", dpi=150)
    plt.close(fig)

    # hour dependence colored by holiday
    hol = X["is_holiday"].to_numpy()
    fig, ax = plt.subplots(figsize=(7, 3.8))
    j = cols.index("hour")
    for v, col, lab in ((0, "#3182bd", "非假日"), (1, "#e6550d", "假日")):
        mm = hol == v
        ax.scatter(X["hour"].to_numpy()[mm] + rng.uniform(-.25, .25, mm.sum()), sv[mm, j], s=3, alpha=.3,
                   color=col, label=lab)
    ax.set_xlabel("目標小時")
    ax.set_ylabel("SHAP(hour)（量級單位）")
    ax.legend(markerscale=4)
    ax.set_title("時段的 SHAP 依賴圖（依假日著色）")
    fig.savefig(FIGURES / "reg_shap_dependence_hour.png", dpi=150)
    plt.close(fig)

    # total holiday/daytype group contribution by hour
    hc = [cols.index(c) for c in imp.feature[imp.group == "假日／日別"]]
    d = pd.DataFrame({"hour": X["hour"].to_numpy(), "hol": hol, "v": svp[:, hc].sum(1)})
    piv = d.groupby(["hour", "hol"])["v"].mean().unstack()
    fig, ax = plt.subplots(figsize=(7, 3.5))
    for v, col, lab in ((0, "#3182bd", "非假日"), (1, "#e6550d", "假日")):
        if v in piv:
            ax.plot(piv.index, piv[v], marker="o", ms=3, color=col, label=lab)
    ax.axhline(0, color="gray", lw=.6)
    ax.set_xlabel("目標小時")
    ax.set_ylabel("假日／日別特徵 SHAP 合計（人次）")
    ax.legend()
    ax.set_title("假日／日別特徵對預測的平均貢獻（依小時）")
    fig.savefig(FIGURES / "reg_shap_holiday_by_hour.png", dpi=150)
    plt.close(fig)

    if "precip" in cols:
        # rainy-hour sample (daytime 7-22 h so the relative effect is not dominated by near-zero night volumes)
        te_d = te[te["hour"].between(7, 22).to_numpy()]
        rain_idx = np.where(te_d["precip"].to_numpy() > 0)[0]
        dry_idx = np.where(te_d["precip"].to_numpy() == 0)[0]
        ridx = np.concatenate([rng.choice(rain_idx, size=min(len(rain_idx), n), replace=False),
                               rng.choice(dry_idx, size=min(len(dry_idx), n // 4), replace=False)])
        Xr = te_d.iloc[ridx][cols]
        svr = expl.shap_values(Xr) * te_d.iloc[ridx]["scale"].to_numpy()[:, None]
        wc = [cols.index(c) for c in cols if group_of(c) == "天氣"]
        tot = svr[:, wc].sum(1)
        base = te_d.iloc[ridx]["hist_dt"].to_numpy()
        dt = np.where(Xr["daytype"].to_numpy() == 0, "平日", "週末／假日")
        pr = Xr["precip"].to_numpy()
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for v, col in (("平日", "#3182bd"), ("週末／假日", "#e6550d")):
            mm = dt == v
            axes[0].scatter(np.minimum(pr[mm], 12) + rng.uniform(-.05, .05, mm.sum()),
                            svr[mm, cols.index("precip")], s=5, alpha=.35, color=col, label=v)
        axes[0].set_xlabel("目標小時降雨 (mm，>12 併入 12)")
        axes[0].set_ylabel("SHAP(precip)（人次）")
        axes[0].set_title("降雨 × 日別：precip 的 SHAP 依賴圖（7–22 時）")
        axes[0].axhline(0, color="gray", lw=.6)
        axes[0].legend(markerscale=3, fontsize=8)
        bins = pd.cut(pd.Series(pr), [-0.01, 0, 1, 3, 1e9], labels=["0", "0.1–1", "1–3", "≥3"])
        tab = pd.DataFrame({"precip_bin": bins.to_numpy(), "daytype": dt, "shap": tot, "base": base}).groupby(
            ["precip_bin", "daytype"], observed=True).agg(n=("shap", "size"), weather_shap_persons=("shap", "mean"),
                                                         base_persons=("base", "mean"))
        tab["weather_shap_pct_of_profile"] = 100 * tab.weather_shap_persons / tab.base_persons
        tab = tab.round(2)
        tab.to_csv(RESULTS / "reg_shap_rain_daytype.csv")
        print(tab)
        piv = tab["weather_shap_pct_of_profile"].unstack()
        x = np.arange(len(piv))
        for i, (v, col) in enumerate((("平日", "#3182bd"), ("週末／假日", "#e6550d"))):
            if v in piv:
                axes[1].bar(x + (i - .5) * .38, piv[v], .38, color=col, label=v)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(piv.index.astype(str))
        axes[1].set_xlabel("目標小時降雨 (mm)")
        axes[1].set_ylabel("天氣特徵 SHAP 合計 / 歷史平均（%）")
        axes[1].set_title("天氣總效果（相對同站同時段的歷史平均）")
        axes[1].axhline(0, color="gray", lw=.6)
        axes[1].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(FIGURES / "reg_shap_dependence_precip_holiday.png", dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    main()
