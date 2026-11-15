"""Dual-threshold crowd definition + class-imbalance strategies (t+1).

Labels (target hour t+1, total = entries + exits):
  relative  total >= station P95 on training data (proposal / Table 2)
  absolute  total >= A, A = pooled P95 of all station-hours in service hours (06-23) on training data
            (~5,900 trips/hour, i.e. ~100 passengers per minute through the faregates)
  dual      relative AND absolute: unusual for the station *and* large in absolute terms
Scores: HistAvg ratio, LightGBM-scaled regression / threshold, LightGBM binary with
  none | class_weight | focal | undersample. Features = base (features.py) + spatial (41).
Threshold: highest score with validation Recall >= 0.7. Target: Precision >= 0.9 at Recall >= 0.7.

Usage: python src/42_crowd_main.py [--quick] [--no-spatial]
Outputs results/clf_main.csv, results/clf_main.md, results/clf_label_dist.csv,
        figures/clf_pr_curves.png, figures/clf_label_dist.png
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

sys.path.insert(0, str(Path(__file__).parent))
from config import FIGURES, PROCESSED, RESULTS  # noqa: E402
from features import load_panel  # noqa: E402
from metrics import clf_metrics, pick_threshold  # noqa: E402

lib = importlib.import_module("40_crowd_lib")
LABELS = {"relative": "label", "absolute": "label_abs", "dual": "label_dual"}
LABEL_ZH = {"relative": "相對門檻（各站 P95）", "absolute": "絕對門檻（全網 P95）", "dual": "雙門檻（相對且絕對）"}


def size_groups(stations) -> pd.Series:
    st = pd.read_csv(PROCESSED / "stations.csv").sort_values("mean_total", ascending=False).reset_index(drop=True)
    g = np.where(st.index < 20, "大站(前20)", np.where(st.index < 60, "中站(21-60)", "小站(61+)"))
    return pd.Series(g, index=st.station)


def add_labels(fr, thr_rel, A):
    fr["crowd_thr"] = fr["station"].map(thr_rel).astype(float)
    fr["label"] = (fr["y"] >= fr["crowd_thr"]).astype(int)
    fr["label_abs"] = (fr["y"] >= A).astype(int)
    fr["label_dual"] = fr["label"] & fr["label_abs"]
    return fr


def ratio_score(pred, fr, label, A):
    r_rel, r_abs = pred / fr["crowd_thr"].to_numpy(), pred / A
    return {"relative": r_rel, "absolute": r_abs, "dual": np.minimum(r_rel, r_abs)}[label]


def safe_metrics(y, s, t):
    if y.sum() == 0 or y.sum() == len(y):
        pred = s >= t
        return {"AUC": np.nan, "AP": np.nan, "Precision": np.nan, "Recall": np.nan, "F1": np.nan, "thr": round(t, 4),
                "pos_rate": round(float(y.mean()), 4), "n_alerts": int(pred.sum())}
    return clf_metrics(y, s, t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="only none/class_weight classifiers")
    ap.add_argument("--no-spatial", action="store_true")
    a = ap.parse_args()
    panel = load_panel()
    thr_rel = lib.crowd_thresholds(panel)
    A = lib.abs_threshold(panel)
    print(f"absolute threshold A = {A:.0f} trips/hour")
    sp = not a.no_spatial
    fr = lib.total_frame(panel, h=1, spatial=sp, cache="tot")
    fr = add_labels(fr, thr_rel, A)
    tr, va, te = lib.split(fr)
    cols = lib.base_cols(fr) + (lib.spatial_cols(fr) if sp else [])
    grp = size_groups(fr.station.cat.categories)

    # label distribution by station size
    dist = []
    for name, col in LABELS.items():
        for part, d in (("train", tr), ("test", te)):
            g = d.groupby(d.station.map(grp).astype(str))[col].agg(["sum", "mean"])
            for k, r in g.iterrows():
                dist.append({"label": name, "split": part, "group": k, "positives": int(r["sum"]),
                             "pos_rate": round(r["mean"], 4)})
            dist.append({"label": name, "split": part, "group": "全部", "positives": int(d[col].sum()),
                         "pos_rate": round(d[col].mean(), 4)})
    dist = pd.DataFrame(dist)
    dist.to_csv(RESULTS / "clf_label_dist.csv", index=False)
    n_st = te.groupby("station", observed=True)[["label", "label_abs", "label_dual"]].sum().gt(0).sum()
    print(dist[dist.split == "test"].to_string(index=False))

    # models
    reg = lib.fit_scaled_reg(tr, va, cols)
    reg_va, reg_te = lib.pred_scaled(reg, va, cols), lib.pred_scaled(reg, te, cols)
    modes = ["none", "class_weight"] if a.quick else ["none", "class_weight", "focal", "undersample"]
    rows, curves = [], {}
    for name, col in LABELS.items():
        S_va = {"HistAvg": ratio_score(va.hist_avg.to_numpy(), va, name, A),
                "LGB-reg/thr": ratio_score(reg_va, va, name, A)}
        S_te = {"HistAvg": ratio_score(te.hist_avg.to_numpy(), te, name, A),
                "LGB-reg/thr": ratio_score(reg_te, te, name, A)}
        for mode in modes:
            c = lib.fit_clf(tr, va, cols, label=col, mode=mode)
            S_va[f"LGB-clf[{mode}]"] = lib.pred_clf(c, va, cols)
            S_te[f"LGB-clf[{mode}]"] = lib.pred_clf(c, te, cols)
            print(name, mode, "iters", c.best_iteration, flush=True)
        yv, yt = va[col].to_numpy(), te[col].to_numpy()
        g_te = te.station.map(grp).astype(str).to_numpy()
        for meth in S_te:
            t = pick_threshold(yv, S_va[meth], 0.7)
            m = safe_metrics(yt, S_te[meth], t)
            rows.append({"label": name, "method": meth, "group": "全部", **m,
                         "target_met": bool(m["Recall"] >= 0.7 and m["Precision"] >= 0.9)})
            for gname in ("大站(前20)", "中站(21-60)", "小站(61+)"):
                mk = g_te == gname
                rows.append({"label": name, "method": meth, "group": gname, **safe_metrics(yt[mk], S_te[meth][mk], t)})
            p, r, _ = precision_recall_curve(yt, S_te[meth])
            curves[(name, meth)] = (p, r)
        print(pd.DataFrame([r for r in rows if r["label"] == name and r["group"] == "全部"]).to_string(index=False),
              flush=True)
    res = pd.DataFrame(rows)
    res.insert(0, "spatial", sp)
    res.to_csv(RESULTS / "clf_main.csv", index=False)

    main_tab = res[res.group == "全部"].drop(columns=["group", "spatial"])
    grp_tab = res[res.group != "全部"].pivot_table(index=["label", "method"], columns="group",
                                                 values=["Precision", "Recall"]).round(3)
    grp_tab.columns = [f"{a}_{b}" for a, b in grp_tab.columns]
    md = ["# 擁擠預警主結果（t+1，測試集 2026-09）", "",
          f"- 特徵：features.py 基礎特徵{' ＋ 空間特徵（41_spatial_features）' if sp else ''}",
          f"- 相對門檻：各站訓練期 (進+出) P95；絕對門檻 A = 全網站時（06–23 時）訓練期 P95 = {A:.0f} 人次/小時；"
          "雙門檻 = 兩者同時成立",
          f"- 測試期出現正類的站數：相對 {n_st['label']}、絕對 {n_st['label_abs']}、雙門檻 {n_st['label_dual']}（共 122 站）",
          "- 決策門檻：驗證集（2026-08）Recall ≥ 0.7 的最高門檻；target_met = 測試集 Recall ≥ 0.7 且 Precision ≥ 0.9",
          "", "## 整體", "", lib.to_md(main_tab), "", "## 正類分布（測試集）", "",
          lib.to_md(dist[dist.split == "test"].drop(columns="split")), "",
          "## 依站點規模（測試集，Precision／Recall）", "", lib.to_md(grp_tab.reset_index()), ""]
    (RESULTS / "clf_main.md").write_text("\n".join(md), encoding="utf-8")

    from plotting import plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    colors = {"HistAvg": "#9aa5b1", "LGB-reg/thr": "#2e9e6a", "LGB-clf[none]": "#5b8def",
              "LGB-clf[class_weight]": "#f2a541", "LGB-clf[focal]": "#c4458f", "LGB-clf[undersample]": "#7a5195"}
    for ax, name in zip(axes, LABELS):
        for meth in [m for (n, m) in curves if n == name]:
            p, r = curves[(name, meth)]
            ap_ = res[(res.label == name) & (res.method == meth) & (res.group == "全部")]["AP"].iloc[0]
            ax.plot(r, p, label=f"{meth} (AP {ap_:.3f})", color=colors.get(meth), lw=1.4)
        ax.axvline(0.7, color="k", ls=":", lw=0.8)
        ax.axhline(0.9, color="k", ls=":", lw=0.8)
        ax.fill_between([0.7, 1], 0.9, 1.0, color="#2e9e6a", alpha=0.08)
        pr = dist[(dist.label == name) & (dist.split == "test") & (dist.group == "全部")].pos_rate.iloc[0]
        ax.set(xlim=(0, 1), ylim=(0.4, 1.01), xlabel="Recall", ylabel="Precision",
               title=f"{LABEL_ZH[name]}，正類率 {pr:.1%}")
        ax.legend(fontsize=7.5, loc="lower left")
    fig.suptitle("擁擠預警 PR 曲線（t+1，測試集 2026-09；綠區＝Recall≥0.7 且 Precision≥0.9）")
    fig.savefig(FIGURES / "clf_pr_curves.png", dpi=150)

    d = dist[dist.split == "test"]
    d = d[d.group != "全部"].pivot(index="group", columns="label", values="positives")
    d = d.reindex(["大站(前20)", "中站(21-60)", "小站(61+)"])[list(LABELS)]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    d.rename(columns=LABEL_ZH).plot.bar(ax=ax, rot=0, color=["#5b8def", "#f2a541", "#2e9e6a"])
    ax.set(ylabel="測試集正類站時數", xlabel="", title="三種擁擠定義的正類分布（依站點規模）")
    fig.savefig(FIGURES / "clf_label_dist.png", dpi=150)


if __name__ == "__main__":
    main()
