"""Alert replay backtest: every hour, for every station, issue t+1 / t+2 crowd warnings and score
them at the level of crowd *events* (consecutive crowded hours of a station merged into one event).

Policy: target hour k of a station is flagged if p1(origin k-1) >= tau or p2(origin k-2) >= tau,
where p1 / p2 are LightGBM classifiers (relative P95 label, base + spatial features) for t+1 / t+2.
Metrics per period
  n_events, prewarn_rate   share of events flagged *before* they start (an alert issued at t <= start-1
                           that targets an hour of the event) -> "detection rate"
  n_nonroutine / prewarn_nonroutine  events whose hours are all below the station threshold by the
                           historical average (i.e. not a recurring peak) and their pre-warning rate
  any_rate                 share of events with any alert targeting one of their hours
  miss_rate                1 - prewarn_rate
  mean_lead_h              mean (event start - earliest issuing time) over pre-warned events (1..2 h)
  fa_per_station_day       flagged target hours that are not crowded, per station per day
  hour_precision / hour_recall  hour-level metrics of the combined policy
Cost: C = ratio * missed events + 1 * false-alarm hours, swept over tau; tau* is chosen on the
validation month and then applied to the test month (no peeking).

Periods: main split (train ~2026-07, valid 2026-08, test 2026-09). Full mode adds rolling origins on
the hist panel (2024-10..): test months 2026-01 (元旦), 2026-02 (春節), 2026-04 (清明連假), 2026-06
(端午) each with train = all data up to month-2, valid = month-1, thresholds/HA/OD re-estimated.

Usage: python src/43_alert_backtest.py [--quick]   (quick = main split only)
Outputs results/clf_backtest.csv, results/clf_backtest_by_cluster.csv, results/clf_backtest_cost.csv, results/clf_backtest_events.parquet,
        figures/clf_backtest_cost.png, figures/clf_backtest_lead.png, figures/clf_backtest_timeline.png
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import FIGURES, RESULTS, TEST_END, TEST_START, TRAIN_END, VALID_END, VALID_START  # noqa: E402
from features import load_panel  # noqa: E402
from metrics import pick_threshold  # noqa: E402

lib = importlib.import_module("40_crowd_lib")
RATIOS = (5, 10, 20)
TAUS = np.round(np.concatenate([np.arange(0.02, 0.2, 0.01), np.arange(0.2, 0.96, 0.025)]), 3)
CLF_MODE = "none"


def month_bounds(ym: str):
    p = pd.Period(ym, "M")
    return str(p.start_time), str(p.end_time.floor("h"))


def fold_scores(panel, train_end, valid, test, od_start_ym=None, cache=None):
    """Return (valid_df, test_df) with columns station, ts_target, label, p1, p2 on the target-hour grid."""
    thr = lib.crowd_thresholds(panel, train_end=train_end)
    out = {}
    for h in (1, 2):
        fr = lib.total_frame(panel, h=h, spatial=True, train_end=train_end, od_start_ym=od_start_ym,
                             cache=cache)
        fr["crowd_thr"] = fr["station"].map(thr).astype(float)
        fr["label"] = (fr["y"] >= fr["crowd_thr"]).astype(int)
        tr, va, te = lib.split(fr, train_end=train_end, valid=valid, test=test)
        cols = lib.base_cols(fr) + lib.spatial_cols(fr)
        m = lib.fit_clf(tr, va, cols, mode=CLF_MODE)
        for part, d in (("valid", va), ("test", te)):
            s = d[["station", "ts_target", "label"]].copy()
            s[f"p{h}"] = lib.pred_clf(m, d, cols)
            if h == 1:   # HistAvg would flag this hour -> "routine" crowding (recurring peak)
                s["routine"] = (d["hist_avg"] >= d["crowd_thr"]).astype(int).to_numpy()
            out[(part, h)] = s
        print(f"  h={h} iters={m.best_iteration} pos(test)={te.label.mean():.4f}", flush=True)
    res = {}
    for part in ("valid", "test"):
        a, b = out[(part, 1)], out[(part, 2)]
        res[part] = a.merge(b[["station", "ts_target", "p2"]], on=["station", "ts_target"], how="inner")
        res[part] = res[part].sort_values(["station", "ts_target"]).reset_index(drop=True)
    return res["valid"], res["test"]


def events_of(df: pd.DataFrame) -> pd.DataFrame:
    """Runs of consecutive crowded hours per station -> (station, start, end, length)."""
    d = df[df.label == 1][["station", "ts_target"]].copy()
    if d.empty:
        return pd.DataFrame(columns=["station", "start", "end", "length"])
    gap = d.groupby("station", observed=True)["ts_target"].diff() != pd.Timedelta(hours=1)
    d["eid"] = gap.cumsum()
    agg = dict(station=("station", "first"), start=("ts_target", "min"), end=("ts_target", "max"),
               length=("ts_target", "size"))
    if "routine" in df:
        d["routine"] = df.loc[d.index, "routine"]
        agg["routine"] = ("routine", "max")
    ev = d.groupby("eid").agg(**agg).reset_index(drop=True)
    return ev


def evaluate(df: pd.DataFrame, tau1: float, tau2: float | None = None, ev: pd.DataFrame | None = None,
             return_events: bool = False):
    tau2 = tau1 if tau2 is None else tau2
    ev = events_of(df) if ev is None else ev
    a1 = df["p1"].to_numpy() >= tau1        # issued at k-1 targeting k
    a2 = df["p2"].to_numpy() >= tau2        # issued at k-2 targeting k
    flag = a1 | a2
    # earliest issuance lead for each target hour: 2 if a2, else 1 if a1, else 0
    lead_k = np.where(a2, 2, np.where(a1, 1, 0))
    # df is sorted by (station, ts_target); locate event start / end rows
    key = pd.MultiIndex.from_arrays([df["station"].astype(str).to_numpy(), df["ts_target"].to_numpy()])
    ps = key.get_indexer(pd.MultiIndex.from_arrays([ev["station"].astype(str).to_numpy(), ev["start"].to_numpy()]))
    pe = key.get_indexer(pd.MultiIndex.from_arrays([ev["station"].astype(str).to_numpy(), ev["end"].to_numpy()]))
    if len(ev):
        assert (ps >= 0).all() and (pe >= 0).all()
    # an alert targeting hour start+j with lead L is issued at start+j-L: pre-warned iff j < L (L <= 2)
    L0 = lead_k[ps] if len(ev) else np.array([], int)
    L1 = np.where(ev["length"].to_numpy() > 1, lead_k[np.minimum(ps + 1, len(df) - 1)], 0) if len(ev) else L0
    leads = np.maximum(L0, np.where(L1 == 2, 1, 0))
    cs = np.concatenate([[0], np.cumsum(flag)])
    anyhit = (cs[pe + 1] - cs[ps]) > 0 if len(ev) else np.array([], bool)
    y = df["label"].to_numpy()
    n_st_day = df["station"].nunique() * df["ts_target"].dt.normalize().nunique()
    fa = int((flag & (y == 0)).sum())
    n_ev = len(ev)
    r = {"tau1": tau1, "tau2": tau2, "n_events": n_ev,
         "prewarn_rate": round(float((leads > 0).mean()), 4) if n_ev else np.nan,
         "any_rate": round(float(anyhit.mean()), 4) if n_ev else np.nan,
         "miss_rate": round(float(1 - (leads > 0).mean()), 4) if n_ev else np.nan,
         "missed_events": int((leads == 0).sum()),
         "mean_lead_h": round(float(leads[leads > 0].mean()), 3) if (leads > 0).any() else np.nan,
         "share_lead2": round(float((leads == 2).mean()), 4) if n_ev else np.nan,
         "n_nonroutine": int((ev["routine"] == 0).sum()) if "routine" in ev else np.nan,
         "prewarn_nonroutine": (round(float((leads[ev["routine"].to_numpy() == 0] > 0).mean()), 4)
                                if "routine" in ev and (ev["routine"] == 0).any() else np.nan),
         "false_alarm_hours": fa, "fa_per_station_day": round(fa / n_st_day, 4),
         "hour_precision": round(float((flag & (y == 1)).sum() / max(flag.sum(), 1)), 4),
         "hour_recall": round(float((flag & (y == 1)).sum() / max(y.sum(), 1)), 4)}
    if return_events:
        e2 = ev.copy()
        e2["lead_h"], e2["any_alert"] = leads, anyhit
        return r, e2
    return r


def run_period(name, va, te, rows, cost_rows, ev_rows):
    ev_va, ev_te = events_of(va), events_of(te)
    sweep = {}
    for part, d, ev in (("valid", va, ev_va), ("test", te, ev_te)):
        for tau in TAUS:
            r = evaluate(d, tau, ev=ev)
            sweep[(part, tau)] = r
            for ratio in RATIOS:
                cost_rows.append({"period": name, "split": part, "tau": tau, "ratio": f"{ratio}:1",
                                  "cost": ratio * r["missed_events"] + r["false_alarm_hours"],
                                  "missed_events": r["missed_events"], "false_alarm_hours": r["false_alarm_hours"]})
    # policy 1: recall>=0.7 thresholds picked on validation separately for each horizon
    t1, t2 = pick_threshold(va.label, va.p1, 0.7), pick_threshold(va.label, va.p2, 0.7)
    r, e = evaluate(te, t1, t2, ev=ev_te, return_events=True)
    rows.append({"period": name, "policy": "hour Recall≥0.7 (valid)", **r})
    e["period"] = name
    ev_rows.append(e)
    # policy 2: min expected cost on validation for each cost ratio
    for ratio in RATIOS:
        costs = {tau: ratio * sweep[("valid", tau)]["missed_events"] + sweep[("valid", tau)]["false_alarm_hours"]
                 for tau in TAUS}
        tau_star = min(costs, key=costs.get)
        rows.append({"period": name, "policy": f"min cost {ratio}:1 (valid)", **evaluate(te, tau_star, ev=ev_te)})
    print(pd.DataFrame([x for x in rows if x["period"] == name]).to_string(index=False), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--plots-only", action="store_true", help="redraw figures from saved results")
    a = ap.parse_args()
    if a.plots_only:
        te = pd.read_parquet(RESULTS / "clf_backtest_scores_2026-09.parquet")
        te["station"] = te["station"].astype("category")
        res = pd.read_csv(RESULTS / "clf_backtest.csv")
        by_cluster(te, res)
        plots(res, pd.read_csv(RESULTS / "clf_backtest_cost.csv"), None, te)
        return
    rows, cost_rows, ev_rows = [], [], []

    print("period 2026-09 (main split)", flush=True)
    panel = load_panel()
    va, te = fold_scores(panel, TRAIN_END, (VALID_START, VALID_END), (TEST_START, TEST_END), cache="tot")
    run_period("2026-09", va, te, rows, cost_rows, ev_rows)
    main_te = te
    te.to_parquet(RESULTS / "clf_backtest_scores_2026-09.parquet", index=False)

    if not a.quick:
        hist = load_panel(hist=True)
        for ym in ("2026-01", "2026-02", "2026-04", "2026-06"):
            p = pd.Period(ym, "M")
            train_end = str((p - 2).end_time.floor("h"))
            print(f"period {ym}: train <= {train_end}", flush=True)
            va, te = fold_scores(hist, train_end, month_bounds(str(p - 1)), month_bounds(ym), od_start_ym="202510")
            run_period(ym, va, te, rows, cost_rows, ev_rows)

    res = pd.DataFrame(rows)
    res.to_csv(RESULTS / "clf_backtest.csv", index=False)
    cost = pd.DataFrame(cost_rows)
    cost.to_csv(RESULTS / "clf_backtest_cost.csv", index=False)
    evs = pd.concat(ev_rows, ignore_index=True)
    evs["station"] = evs["station"].astype(str)
    evs.to_parquet(RESULTS / "clf_backtest_events.parquet", index=False)
    (RESULTS / "clf_backtest.md").write_text(
        "# 預警回放回測（事件層級）\n\n" + lib.to_md(res) + "\n", encoding="utf-8")
    by_cluster(main_te, res)
    plots(res, cost, evs, main_te)


def by_cluster(te: pd.DataFrame, res: pd.DataFrame) -> pd.DataFrame | None:
    """Main-period backtest broken down by workstream C's station types (results/cluster_assignments.csv)."""
    f = RESULTS / "cluster_assignments.csv"
    if not f.exists():
        return None
    cl = pd.read_csv(f).set_index("station")["cluster_name"]
    rows = []
    for pol in ("hour Recall≥0.7 (valid)", "min cost 10:1 (valid)"):
        r = res[(res.period == "2026-09") & (res.policy == pol)].iloc[0]
        for name, d in te.groupby(te["station"].astype(str).map(cl).fillna("未分群"), sort=True):
            d = d.sort_values(["station", "ts_target"]).reset_index(drop=True)
            rows.append({"policy": pol, "cluster": name, "n_stations": d.station.nunique(),
                         **evaluate(d, r.tau1, r.tau2)})
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "clf_backtest_by_cluster.csv", index=False)
    return out


def plots(res, cost, evs, te):
    from plotting import plt
    sys.path.insert(0, str(Path(__file__).parent))
    # cost-threshold curves (main period)
    fig, axes = plt.subplots(1, len(RATIOS), figsize=(14, 3.8))
    for ax, ratio in zip(axes, RATIOS):
        for part, ls in (("valid", "--"), ("test", "-")):
            d = cost[(cost.period == "2026-09") & (cost.split == part) & (cost.ratio == f"{ratio}:1")]
            ax.plot(d.tau, d.cost, ls, label="驗證 2026-08" if part == "valid" else "測試 2026-09")
            if part == "valid":
                ts = d.loc[d.cost.idxmin(), "tau"]
                ax.axvline(ts, color="k", ls=":", lw=0.8)
                ax.text(ts, d.cost.max() * 0.9, f" τ*={ts:.2f}", fontsize=8)
        ax.set(xlabel="預警機率門檻 τ", title=f"漏報:誤報 = {ratio}:1")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("成本＝比例×漏報事件＋誤報小時")
    fig.suptitle("成本－門檻曲線（τ* 於驗證月選定）", y=1.03)
    fig.savefig(FIGURES / "clf_backtest_cost.png", dpi=150)

    # detection / lead / false alarms vs tau (main test) + per period summary
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    d = cost[(cost.period == "2026-09") & (cost.split == "test") & (cost.ratio == "10:1")]
    ev_main = events_of(te)
    sw = pd.DataFrame([evaluate(te, t, ev=ev_main) for t in TAUS])
    ax = axes[0]
    ax.plot(sw.tau1, sw.prewarn_rate, label="事件提前偵測率")
    ax.plot(sw.tau1, sw.hour_precision, label="小時層級 Precision")
    ax2 = ax.twinx()
    ax2.plot(sw.tau1, sw.fa_per_station_day, color="#c4458f", label="誤報／站／日")
    ax2.set_ylabel("誤報／站／日")
    ax.set(xlabel="τ", ylabel="比例", title="測試期 2026-09：門檻取捨")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="center left")
    ax = axes[1]
    pol = res[res.policy == "min cost 10:1 (valid)"]
    x = np.arange(len(pol))
    ax.bar(x - 0.2, pol.prewarn_rate, 0.4, label="提前偵測率")
    ax.bar(x + 0.2, pol.hour_precision, 0.4, label="小時 Precision")
    for i, r in enumerate(pol.itertuples()):
        ax.text(i, 1.02, f"提前 {r.mean_lead_h:.2f}h\n誤報 {r.fa_per_station_day:.2f}/站日", ha="center",
                fontsize=7.5)
    ax.set_xticks(x, pol.period)
    ax.set(ylim=(0, 1.18), title="各回測月（成本比 10:1 的 τ*）")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2)
    fig.savefig(FIGURES / "clf_backtest_lead.png", dpi=150)

    # timeline example: busiest event station in the main test month
    tau = res[(res.period == "2026-09") & (res.policy == "min cost 10:1 (valid)")].tau1.iloc[0]
    st = te.groupby("station", observed=True)["label"].sum().idxmax()
    d = te[te.station == st].set_index("ts_target")
    d = d.loc[d.index[0]: d.index[0] + pd.Timedelta(days=7)]
    fig, ax = plt.subplots(figsize=(13, 3.6))
    ax.plot(d.index, d.p1, label="p(t+1)", lw=1)
    ax.plot(d.index, d.p2, label="p(t+2)", lw=1, alpha=0.8)
    ax.fill_between(d.index, 0, 1, where=d.label == 1, color="#f2a541", alpha=0.3, step="mid", label="實際擁擠")
    ax.axhline(tau, color="k", ls=":", lw=0.8, label=f"τ*={tau:.2f}")
    ax.set(ylim=(0, 1), title=f"預警回放範例：{st}（2026-09 第一週）", ylabel="擁擠機率")
    ax.legend(fontsize=8, ncol=4, loc="upper right")
    fig.savefig(FIGURES / "clf_backtest_timeline.png", dpi=150)


if __name__ == "__main__":
    main()
