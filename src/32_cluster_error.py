"""Workstream C-3: forecast error by station type (cluster).

Inputs: results/cluster_assignments.csv (src/31), results/pred_test_h{h}_{target}.parquet (src/04),
        results/pred_ts_test_h{h}_{target}.parquet (src/30; optional - TS methods skipped if missing).
Metrics per cluster x target x horizon x method (test month 2026-09):
  MAE, RMSE, weekday-peak MAE, relative MAE = MAE / mean(y), relative peak MAE = peak MAE / mean(y at peak),
  rainy-hour MAE (precip >= 1 mm), dry MAE, rainy relative MAE, rain/dry relative-MAE ratio,
  and LightGBM-scaled improvement over Historical Average.
Usage: python src/32_cluster_error.py [--horizons 1 2]
Outputs: results/cluster_error.csv, results/cluster_error.md, figures/cluster_error.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import FIGURES, RAIN_MM, RESULTS  # noqa: E402
from metrics import peak_mask  # noqa: E402
from plotting import plt  # noqa: E402

METHODS = ["Seasonal Naive", "Historical Average", "SARIMAX", "Prophet", "Prophet+AR", "LightGBM", "LightGBM (scaled)"]
ALL = "全部站點"


def load(h: int, target: str) -> pd.DataFrame | None:
    f = RESULTS / f"pred_test_h{h}_{target}.parquet"
    if not f.exists():
        return None
    d = pd.read_parquet(f)
    d["station"] = d.station.astype(str)
    g = RESULTS / f"pred_ts_test_h{h}_{target}.parquet"
    if g.exists():
        t = pd.read_parquet(g).drop(columns="y")
        d = d.merge(t, on=["ts_target", "station"], how="left")
    return d


def group_metrics(d: pd.DataFrame, methods) -> dict:
    y = d.y.to_numpy()
    pk = peak_mask(d)
    rain = (d.precip >= RAIN_MM).to_numpy() if "precip" in d else np.zeros(len(d), bool)
    out = {}
    for m in methods:
        if m not in d or d[m].isna().all():
            continue
        e = np.abs(d[m].to_numpy() - y)
        r = {"MAE": e.mean(), "RMSE": np.sqrt(((d[m].to_numpy() - y) ** 2).mean()), "peak_MAE": e[pk].mean(),
             "rel_MAE": e.mean() / y.mean(), "rel_peak_MAE": e[pk].mean() / y[pk].mean()}
        if rain.any():
            r.update(rain_MAE=e[rain].mean(), dry_MAE=e[~rain].mean(),
                     rel_rain_MAE=e[rain].mean() / y[rain].mean(), rel_dry_MAE=e[~rain].mean() / y[~rain].mean())
            r["rain_dry_ratio"] = r["rel_rain_MAE"] / r["rel_dry_MAE"]
        out[m] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizons", nargs="+", type=int, default=[1, 2])
    a = ap.parse_args()
    asg = pd.read_csv(RESULTS / "cluster_assignments.csv")
    cname = asg.set_index("station").cluster_name
    order = asg.sort_values("cluster").cluster_name.drop_duplicates().tolist()
    rows = []
    for h in a.horizons:
        for target in ("entries", "exits"):
            d = load(h, target)
            if d is None:
                print(f"skip h={h} {target}: pred_test_h{h}_{target}.parquet missing")
                continue
            d["cluster_name"] = d.station.map(cname)
            groups = [(ALL, d)] + [(c, d[d.cluster_name == c]) for c in order]
            for c, g in groups:
                if g.y.mean() < 1:          # exits of entry-only Y-line gates: identically zero
                    continue
                for m, r in group_metrics(g, METHODS).items():
                    rows.append({"horizon": h, "target": target, "cluster_name": c, "n_stations": g.station.nunique(),
                                 "mean_y": g.y.mean(), "rain_hours": int((g.precip >= RAIN_MM).sum() / g.station.nunique())
                                 if "precip" in g else 0, "method": m, **r})
    res = pd.DataFrame(rows)
    base = res[res.method == "Historical Average"].set_index(["horizon", "target", "cluster_name"]).MAE
    res["impr_vs_HA_%"] = (1 - res.MAE / res.set_index(["horizon", "target", "cluster_name"]).index.map(base)) * 100
    res = res.round(4)
    res.to_csv(RESULTS / "cluster_error.csv", index=False)
    write_md(res, order)
    plot(res, order)
    show = res[(res.horizon == 1) & res.method.isin(["Historical Average", "SARIMAX", "Prophet+AR", "LightGBM (scaled)"])]
    print(show.pivot_table(index=["target", "cluster_name"], columns="method", values="rel_MAE").round(3).to_string())


def write_md(res: pd.DataFrame, order):
    L = ["# 各站型預測誤差（測試集 2026-09）", "",
         "相對 MAE = MAE ÷ 該群平均每小時量；雨天 = 目標小時降雨 ≥ 1 mm。環狀線轉乘口只有進站資料，出站不列入。", ""]
    for h in sorted(res.horizon.unique()):
        for tg in ("entries", "exits"):
            g = res[(res.horizon == h) & (res.target == tg)]
            if g.empty:
                continue
            for metric, ttl in (("rel_MAE", "相對 MAE"), ("rel_peak_MAE", "平日尖峰相對 MAE"),
                                ("rel_rain_MAE", "雨天相對 MAE"), ("rain_dry_ratio", "雨天／無雨 相對 MAE 比")):
                if metric not in g:
                    continue
                pv = g.pivot_table(index="cluster_name", columns="method", values=metric)
                pv = pv.reindex([ALL] + [c for c in order if c in pv.index])[[m for m in METHODS if m in pv]]
                L += [f"## t+{h} {'進站' if tg == 'entries' else '出站'}：{ttl}", "",
                      "| 站型 | " + " | ".join(pv.columns) + " |", "|---|" + "---:|" * len(pv.columns)]
                L += ["| " + i + " | " + " | ".join(f"{v:.3f}" for v in r) + " |" for i, r in pv.iterrows()]
                L.append("")
            base = g[g.method == "LightGBM (scaled)"].set_index("cluster_name").reindex([ALL] + order).dropna(how="all")
            L += [f"## t+{h} {'進站' if tg == 'entries' else '出站'}：LightGBM-scaled 各站型絕對誤差", "",
                  "| 站型 | 站數 | 平均量 | MAE | 尖峰 MAE | 雨天 MAE | 無雨 MAE | 較 HA 改善% |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
            for i, r in base.iterrows():
                L.append(f"| {i} | {int(r.n_stations)} | {r.mean_y:.0f} | {r.MAE:.1f} | {r.peak_MAE:.1f} | "
                         f"{r.get('rain_MAE', np.nan):.1f} | {r.get('dry_MAE', np.nan):.1f} | {r['impr_vs_HA_%']:.1f} |")
            L.append("")
    (RESULTS / "cluster_error.md").write_text("\n".join(L), encoding="utf-8")


def plot(res: pd.DataFrame, order):
    h1 = res[res.horizon == 1]
    methods = [m for m in METHODS if m in h1.method.unique() and m != "LightGBM"]
    colors = dict(zip(METHODS, ["#9e9e9e", "#5d6d7e", "tab:blue", "#f5b041", "tab:orange", "#c39bd3", "tab:red"]))
    panels = [("entries", "rel_MAE", "進站 相對 MAE"), ("exits", "rel_MAE", "出站 相對 MAE"),
              ("entries", "rel_peak_MAE", "進站 平日尖峰 相對 MAE"), ("entries", "rel_rain_MAE", "進站 雨天 相對 MAE")]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), sharey=False)
    for ax, (tg, metric, ttl) in zip(axes.flat, panels):
        g = h1[h1.target == tg]
        if metric not in g:
            ax.set_visible(False)
            continue
        cl = [c for c in order if c in g.cluster_name.unique()]
        x = np.arange(len(cl))
        w = 0.8 / len(methods)
        for i, m in enumerate(methods):
            v = g[g.method == m].set_index("cluster_name").reindex(cl)[metric]
            ax.bar(x + (i - (len(methods) - 1) / 2) * w, v, w, color=colors[m], label=m)
        allv = g[(g.cluster_name == ALL) & (g.method == "LightGBM (scaled)")][metric]
        if len(allv):
            ax.axhline(float(allv.iloc[0]), color="tab:red", ls=":", lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels([c.replace("（", "\n（") for c in cl], fontsize=8)
        ax.set_title(ttl)
        ax.grid(axis="y", alpha=0.3)
    axes.flat[0].legend(fontsize=7, ncol=2)
    fig.suptitle("各站型 t+1 預測誤差（MAE ÷ 平均量；紅色虛線＝LightGBM-scaled 全部站點）")
    fig.tight_layout()
    fig.savefig(FIGURES / "cluster_error.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
