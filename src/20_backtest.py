"""Workstream B task 6: multi-month rolling-origin backtest on the 2024-10..2026-09 panel (119 stations).

Each test month M in 2025-10..2026-09 is predicted by a model trained only on data before M:
  * retrain every `--retrain` months (default 3 = quarterly; 1 = monthly);
  * for a block starting at month B: train on ts_target < B - 1 month, early-stop on month B-1,
    historical-average profiles use data < B only (no leakage), then predict every month in the block.
Leakage: hist_avg (and B's day-type profiles) are recomputed for every block from data < block start
(`20_reglib.add_features(hist_end=...)`, equivalent to make_features(train_end=...)).
Methods: Seasonal Naive, Historical Average (same cut-off), LightGBM-scaled (report version) and LightGBM-B.
Also reports special-day errors (New Year's Eve/Day, Lunar New Year, long weekends, typhoon days if the
external calendar provides them, rainy hours if weather is available).

Usage: python src/20_backtest.py [--quick] [--retrain 3] [--target entries]
  --quick : entries only, LightGBM-B only, learning_rate 0.1, retrain every 6 months.
Writes results/reg_backtest_monthly.csv, results/reg_backtest_special.csv, results/reg_backtest.md,
       figures/reg_backtest_monthly.png
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import EXTERNAL, FIGURES, RAIN_MM, RESULTS  # noqa: E402
from features import load_calendar, load_panel  # noqa: E402
from metrics import reg_metrics  # noqa: E402
from plotting import plt  # noqa: E402

R = importlib.import_module("20_reglib")
TEST_MONTHS = pd.period_range("2025-10", "2026-09", freq="M")


def typhoon_dates() -> set:
    """Typhoon-day closures if workstream A provides them (any csv in data/external with typhoon info)."""
    out = set()
    for name in ("typhoon_days.csv", "typhoon.csv"):
        f = EXTERNAL / name
        if f.exists():
            t = pd.read_csv(f)
            c = [c for c in t.columns if "date" in c.lower()][0]
            out |= set(pd.to_datetime(t[c]).dt.normalize())
    cal = pd.read_csv(EXTERNAL / "calendar.csv", parse_dates=["date"])
    for c in cal.columns:
        if "typhoon" in c.lower():
            out |= set(cal.loc[cal[c].astype(bool), "date"])
    if "holiday_name" in cal:
        out |= set(cal.loc[cal["holiday_name"].fillna("").str.contains("颱風"), "date"])
    return out


def special_labels(ts: pd.Series) -> pd.DataFrame:
    cal = load_calendar().set_index("date")
    d = ts.dt.normalize()
    name = d.map(cal["holiday_name"]).fillna("")
    lab = pd.DataFrame(index=ts.index)
    lab["跨年（12/31 18時–1/1）"] = ((ts.dt.month == 12) & (ts.dt.day == 31) & (ts.dt.hour >= 18)) | \
        ((ts.dt.month == 1) & (ts.dt.day == 1))
    lab["春節連假"] = d.map(cal["holiday_block_len"]).fillna(0).ge(7) & (ts.dt.month.isin([1, 2]))
    lab["連假（非春節）"] = d.map(cal["is_long_weekend"]).fillna(False).astype(bool) & ~lab["春節連假"]
    lab["補班日"] = d.map(cal["is_makeup_workday"]).fillna(False).astype(bool)
    lab["連假前一日"] = (d.map(cal["days_to_next_holiday"]).fillna(99).eq(1)
                    & ~d.map(cal["is_holiday"]).fillna(False).astype(bool))
    ty = typhoon_dates()
    if ty:
        lab["颱風假"] = d.isin(list(ty))
    lab["一般平日"] = (~d.map(cal["is_holiday"]).fillna(False).astype(bool)) & (~lab.drop(columns=[]).any(axis=1))
    del name
    return lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--retrain", type=int, default=3, help="retrain every N months")
    ap.add_argument("--targets", default="entries,exits")
    ap.add_argument("--horizon", type=int, default=1)
    a = ap.parse_args()
    retrain = 6 if a.quick else a.retrain
    targets = ["entries"] if a.quick else a.targets.split(",")
    params = {"learning_rate": 0.1} if a.quick else {"learning_rate": 0.08}
    h = a.horizon
    panel = load_panel(hist=True)
    w = R.load_weather("obs")
    if w is not None:
        print(f"weather covers {w.ts.min()} .. {w.ts.max()}", flush=True)
    rows, sp_rows, allp = [], [], []
    for tgt in targets:
        fr0 = R.make_features(panel, target=tgt, horizon=h)
        for bi in range(0, len(TEST_MONTHS), retrain):
            block = TEST_MONTHS[bi:bi + retrain]
            B = block[0].start_time
            vstart = (block[0] - 1).start_time
            fr = R.add_features(fr0, panel, tgt, h, hist_end=B - pd.Timedelta(hours=1))
            fr = fr.dropna(subset=["y", "lag_336"])
            t = fr["ts_target"]
            tr = fr[t < vstart]
            va = fr[(t >= vstart) & (t < B)]
            te = fr[(t >= B) & (t <= block[-1].end_time)]
            print(f"{tgt} block {block[0]}..{block[-1]}: train {len(tr)} rows ({tr.ts_target.min():%Y-%m}.."
                  f"{tr.ts_target.max():%Y-%m}) valid {vstart:%Y-%m}", flush=True)
            preds = te[["ts_target", "station", "y", "hour", "is_holiday"] +
                       [c for c in ("precip",) if c in te]].copy()
            preds["target"] = tgt
            preds["Seasonal Naive"] = te["lag_168"].to_numpy()
            preds["Historical Average"] = te["hist_avg"].to_numpy()
            if not a.quick:
                cs = R.cols_scaled(fr)
                m = R.fit(tr, va, cs, "regression", True, params=params)
                preds["LightGBM-scaled"] = R.predict(m, te, cs, True)
            cf = R.cols_full(fr)
            m = R.fit(tr, va, cf, R.B_OBJ, True, params=params)
            preds["LightGBM-B"] = R.predict(m, te, cf, True)
            print(f"   LightGBM-B iter={m.best_iteration}", flush=True)
            allp.append(preds)
    P = pd.concat(allp, ignore_index=True)
    methods = [c for c in ("Seasonal Naive", "Historical Average", "LightGBM-scaled", "LightGBM-B") if c in P]
    P["month"] = P["ts_target"].dt.to_period("M").astype(str)
    ty = typhoon_dates()
    P["typhoon"] = P["ts_target"].dt.normalize().isin(list(ty))
    for (tgt, mo), g in P.groupby(["target", "month"]):
        gx = g[~g["typhoon"]]
        for mt in methods:
            r = {"target": tgt, "month": mo, "method": mt, "n": len(g), **reg_metrics(g.y, g[mt], g),
                 "typhoon_hours": int(g["typhoon"].sum() // max(g.station.nunique(), 1)),
                 "MAE_excl_typhoon": round(float(np.abs(gx[mt] - gx.y).mean()), 2)}
            rows.append(r)
    lab = special_labels(P["ts_target"])
    if "precip" in P:
        lab["雨天小時（≥1mm）"] = P["precip"].fillna(0).ge(RAIN_MM)
    for tgt in targets:
        for c in lab.columns:
            msk = lab[c].to_numpy() & (P["target"] == tgt).to_numpy()
            if msk.sum() == 0:
                continue
            g = P[msk]
            r = {"target": tgt, "subset": c, "n": int(msk.sum()), "days": int(g.ts_target.dt.normalize().nunique())}
            for mt in methods:
                r[f"MAE_{mt}"] = round(float(np.abs(g[mt] - g.y).mean()), 2)
                r[f"bias_{mt}"] = round(float((g[mt] - g.y).mean()), 2)
            r["mean_y"] = round(float(g.y.mean()), 1)
            sp_rows.append(r)
    df, sp = pd.DataFrame(rows), pd.DataFrame(sp_rows)
    suffix = "_quick" if a.quick else ""
    df.to_csv(RESULTS / f"reg_backtest_monthly{suffix}.csv", index=False)
    sp.to_csv(RESULTS / f"reg_backtest_special{suffix}.csv", index=False)
    # worst days for LightGBM-B
    P["date"] = P["ts_target"].dt.normalize()
    P["ae"] = (P["LightGBM-B"] - P.y).abs()
    P["e"] = P["LightGBM-B"] - P.y
    worst = P.groupby(["target", "date"]).agg(MAE=("ae", "mean"), bias=("e", "mean")).reset_index()
    cal = load_calendar().set_index("date")
    worst["holiday_name"] = worst["date"].map(cal["holiday_name"]).fillna("")
    worst["dow"] = worst["date"].dt.day_name().str[:3]
    worst = worst.sort_values("MAE", ascending=False).groupby("target").head(12)
    worst.to_csv(RESULTS / f"reg_backtest_worstdays{suffix}.csv", index=False)

    # figure
    fig, axes = plt.subplots(len(targets), 1, figsize=(10, 3.4 * len(targets)), squeeze=False)
    colors = {"Seasonal Naive": "#bdbdbd", "Historical Average": "#fd8d3c", "LightGBM-scaled": "#6baed6",
              "LightGBM-B": "#08519c"}
    for ax, tgt in zip(axes[:, 0], targets):
        d = df[df.target == tgt].pivot(index="month", columns="method", values="MAE")
        x = np.arange(len(d))
        wbar = 0.8 / len(methods)
        for i, mt in enumerate(methods):
            ax.bar(x + (i - (len(methods) - 1) / 2) * wbar, d[mt], wbar, label=mt, color=colors[mt])
        ax.set_xticks(x)
        ax.set_xticklabels([m[2:] for m in d.index], fontsize=8)
        ax.set_ylabel("MAE（人次）")
        ax.set_title(f"{'進' if tgt == 'entries' else '出'}站 t+{h} 逐月滾動回測 MAE（每 {retrain} 個月重訓）")
        ax.legend(fontsize=8, ncol=len(methods))
    fig.tight_layout()
    fig.savefig(FIGURES / f"reg_backtest_monthly{suffix}.png", dpi=150)
    plt.close(fig)

    md = [f"# 多月滾動起點回測（2025-10 ~ 2026-09，119 站，t+{h}，每 {retrain} 個月重訓）\n",
          "每個測試區塊只用區塊開始前的資料訓練（前一個月作 early stopping），歷史平均也只用區塊前資料。"
          "颱風停班日（data/external/typhoon_days.csv）的誤差另列，「排除颱風日」欄為不含這些日子的 MAE。\n",
          "## 各月 MAE\n"]
    for tgt in targets:
        d = df[df.target == tgt].pivot(index="month", columns="method", values="MAE")[methods].reset_index()
        d["B_vs_HA_%"] = (100 * (1 - d["LightGBM-B"] / d["Historical Average"])).round(1)
        dx = df[df.target == tgt].pivot(index="month", columns="method", values="MAE_excl_typhoon")
        d["B 排除颱風日"] = dx["LightGBM-B"].to_numpy()
        d["HA 排除颱風日"] = dx["Historical Average"].to_numpy()
        md += [f"### {tgt}\n", R.md_table(d), ""]
        tot = P[P.target == tgt]
        md.append("全期 MAE：" + "、".join(f"{mt} {np.abs(tot[mt] - tot.y).mean():.1f}" for mt in methods) + "\n")
        tx = tot[~tot["typhoon"]]
        md.append("全期 MAE（排除颱風停班日）：" + "、".join(f"{mt} {np.abs(tx[mt] - tx.y).mean():.1f}"
                                                   for mt in methods) + "\n")
    md += ["## 特殊日誤差（MAE／平均偏差 bias = 預測 − 實際）\n", R.md_table(sp), "",
           "## LightGBM-B 誤差最大的日子\n", R.md_table(worst.assign(date=worst.date.dt.strftime("%Y-%m-%d")))]
    (RESULTS / f"reg_backtest{suffix}.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
