"""EDA for workstream A: hourly profiles, rain effect (with CIs), events, typhoon days.

Reproduces / extends proposal section 3.1.
Outputs: figures/eda_*.png, results/eda_*.csv, results/eda_summary_numbers.json
Usage: python src/11_eda.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import EXTERNAL, FIGURES, MAIN_START, RESULTS  # noqa: E402
from features import load_calendar, load_panel, load_weather  # noqa: E402
from plotting import plt  # noqa: E402

RNG = np.random.default_rng(0)
SERVICE_HOURS = list(range(6, 24))          # exclude 0-5 (little or no service)
# same bins as the progress report (reference/progress_report/src/03_eda.py): 0, (0,1], (1,5], >5 mm
RAIN_BINS = [-1, 0, 1.0, 5.0, np.inf]
RAIN_LABELS = ["無雨", "<1mm", "1–5mm", ">5mm"]
DAYTYPE = {0: "平日", 1: "假日"}
NUM = {}


def network_hourly(panel: pd.DataFrame, cal: pd.DataFrame) -> pd.DataFrame:
    net = panel.groupby("ts", observed=True)[["entries", "exits"]].sum().reset_index()
    net["date"] = net.ts.dt.normalize()
    net["hour"] = net.ts.dt.hour
    net = net.merge(cal[["date", "is_holiday"]], on="date", how="left")
    net["hol"] = net["is_holiday"].fillna(False).astype(int)
    net["ym"] = net.ts.dt.to_period("M").astype(str)
    return net


def boot_ci(values: np.ndarray, groups: np.ndarray, n: int = 1000) -> tuple[float, float]:
    """Cluster bootstrap (resample days) CI of the mean."""
    if len(values) == 0:
        return np.nan, np.nan
    ug, inv = np.unique(groups, return_inverse=True)
    sums = np.bincount(inv, weights=values)
    cnts = np.bincount(inv)
    k = len(ug)
    idx = RNG.integers(0, k, size=(n, k))
    est = sums[idx].sum(1) / cnts[idx].sum(1)
    return tuple(np.percentile(est, [2.5, 97.5]))


# --------------------------------------------------------------------------- 1. hourly profile
def fig_profile(net: pd.DataFrame, main: pd.DataFrame):
    prof = main.groupby(["hol", "hour"])["entries"].mean().unstack(0).rename(columns=DAYTYPE)
    prof.to_csv(RESULTS / "eda_hourly_profile.csv", float_format="%.0f")
    fig, ax = plt.subplots(figsize=(7, 3.4))
    for c, col in zip(prof.columns, ["#1f77b4", "#d62728"]):
        ax.plot(prof.index, prof[c] / 1e3, marker="o", ms=3, label=c, color=col)
    ax.set_xlabel("小時")
    ax.set_ylabel("全網進站量（千人次／小時）")
    ax.set_xticks(range(0, 24, 2))
    ax.set_title("全網平均每小時進站量（2025/10–2026/9）")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig(FIGURES / "eda_hourly_profile.png", dpi=150)
    plt.close(fig)
    daily = main.groupby(["date", "hol"])["entries"].sum().reset_index().groupby("hol")["entries"].mean()
    NUM["daily_entries_workday"] = round(daily.get(0, np.nan))
    NUM["daily_entries_holiday"] = round(daily.get(1, np.nan))
    NUM["holiday_to_workday_ratio"] = round(daily.get(1) / daily.get(0), 3)
    NUM["workday_peak_hours"] = prof["平日"].nlargest(3).index.tolist()


# --------------------------------------------------------------------------- 2. rain
def rain_effect(net: pd.DataFrame, weather: pd.DataFrame, period: str, precip_col: str = "precip"):
    df = net.merge(weather[["ts", precip_col]], on="ts", how="inner")
    df = df[df.hour.isin(SERVICE_HOURS)].copy()
    df["rain"] = pd.cut(df[precip_col], RAIN_BINS, labels=RAIN_LABELS)
    # baseline = mean of no-rain hours in the same (day-type, hour, month)  -> controls season
    dry = df[df["rain"] == RAIN_LABELS[0]]
    base = dry.groupby(["hol", "hour", "ym"])["entries"].mean().rename("base")
    df = df.join(base, on=["hol", "hour", "ym"])
    # proposal-style baseline: same (day-type, hour) only
    base2 = dry.groupby(["hol", "hour"])["entries"].mean().rename("base2")
    df = df.join(base2, on=["hol", "hour"])
    df["ratio"] = df["entries"] / df["base"]
    df["ratio2"] = df["entries"] / df["base2"]
    rows = []
    for (h, r), g in df.dropna(subset=["ratio"]).groupby(["hol", "rain"], observed=True):
        # ratio_mean / ci: proposal definition (baseline = dry hours of same day-type x hour)
        # ratio_month_ctrl: baseline additionally within the same month (controls seasonality)
        lo, hi = boot_ci(g["ratio2"].to_numpy(), g["date"].to_numpy())
        lo2, hi2 = boot_ci(g["ratio"].to_numpy(), g["date"].to_numpy())
        rows.append({"period": period, "day_type": DAYTYPE[h], "rain": r, "n_hours": len(g),
                     "n_days": g["date"].nunique(), "ratio_mean": g["ratio2"].mean(),
                     "ci_low": lo, "ci_high": hi, "ratio_month_ctrl": g["ratio"].mean(),
                     "ci_low_month_ctrl": lo2, "ci_high_month_ctrl": hi2})
    return pd.DataFrame(rows), df


def fig_rain(tab: pd.DataFrame, fname: str, title: str):
    fig, ax = plt.subplots(figsize=(7, 3.6))
    w = 0.38
    x = np.arange(len(RAIN_LABELS))
    for i, (dt, col) in enumerate([("平日", "#1f77b4"), ("假日", "#d62728")]):
        t = tab[tab.day_type == dt].set_index("rain").reindex(RAIN_LABELS)
        err = np.vstack([t.ratio_mean - t.ci_low, t.ci_high - t.ratio_mean]) * 100
        ax.bar(x + (i - 0.5) * w, t.ratio_mean * 100, w, yerr=err, capsize=3, color=col, label=dt, alpha=0.85)
        for xi, (v, n) in enumerate(zip(t.ratio_mean, t.n_hours)):
            if pd.notna(v):
                ax.text(xi + (i - 0.5) * w, 3, f"n={int(n)}", ha="center", va="bottom", fontsize=7,
                        color="white", rotation=90)
    ax.axhline(100, color="k", lw=0.8, ls="--")
    ax.set_xticks(x, RAIN_LABELS)
    ax.set_ylabel("全網進站量／無雨基準（%）")
    ax.set_xlabel("該小時降雨量")
    ax.set_title(title)
    ax.legend()
    fig.savefig(FIGURES / fname, dpi=150)
    plt.close(fig)


def rain_by_hour_heat(df: pd.DataFrame):
    """ratio by hour for rainy (>=1mm) hours, weekday vs holiday."""
    d = df[df["precip"] >= 1.0]
    t = d.groupby(["hol", "hour"])["ratio"].agg(["mean", "count"]).reset_index()
    t["day_type"] = t.hol.map(DAYTYPE)
    t.drop(columns="hol").to_csv(RESULTS / "eda_rain_by_hour.csv", index=False, float_format="%.3f")
    fig, ax = plt.subplots(figsize=(7, 3.2))
    for h, col in [(0, "#1f77b4"), (1, "#d62728")]:
        s = t[(t.hol == h) & (t["count"] >= 5)]
        ax.plot(s.hour, s["mean"] * 100, marker="o", ms=3, color=col, label=DAYTYPE[h])
    ax.axhline(100, color="k", lw=0.8, ls="--")
    ax.set_xlabel("小時")
    ax.set_ylabel("雨天(≥1mm)／無雨基準（%）")
    ax.set_title("雨天人流比例的時段差異（樣本 ≥5 小時）")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig(FIGURES / "eda_rain_by_hour.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- 3. events
def event_effect(panel: pd.DataFrame, cal: pd.DataFrame):
    evf = EXTERNAL / "events.csv"
    if not evf.exists():
        print("events.csv missing -> skip event EDA")
        return
    from importlib import import_module
    evmod = import_module("10_events")
    vmap = evmod.VENUE_STATIONS
    vmap_st = set(panel.station.cat.categories)
    ev = pd.read_csv(evf, dtype=str)
    ev = ev[ev.venue.isin(vmap)].drop_duplicates(["venue", "date"])
    esh = pd.read_csv(EXTERNAL / "events_station_hour.csv", parse_dates=["ts"])
    p = panel.copy()
    p["date"] = p.ts.dt.normalize()
    p["hour"] = p.ts.dt.hour
    p = p.merge(cal[["date", "is_holiday"]], on="date", how="left")
    p["hol"] = p.is_holiday.fillna(False).astype(int)
    rows, curves, cands = [], {}, []
    for venue, g in ev.groupby("venue"):
        st = vmap[venue][0][0]
        days = pd.to_datetime(g["date"].unique())
        ps = p[p.station == st]
        # baseline excludes every day on which any event touches this station (events_station_hour)
        other = esh.loc[esh.station == st, "ts"].dt.normalize().unique()
        base = ps[~ps.date.isin(other)].groupby(["hol", "hour", ps.ts.dt.to_period("Q")])[
            ["entries", "exits"]].mean()
        e = ps[ps.date.isin(days)].copy()
        e["q"] = e.ts.dt.to_period("Q")
        e = e.join(base, on=["hol", "hour", "q"], rsuffix="_base")
        e["r_ent"] = e.entries / e.entries_base
        e["r_exi"] = e.exits / e.exits_base
        # arrival / departure windows from the actual timestamps of 10_events.expand (handles midnight)
        imp = evmod.expand(g.assign(stations=""), set(vmap_st))
        imp = imp[imp.station == st].set_index("ts")
        pts = ps.assign(q=ps.ts.dt.to_period("Q")).join(base, on=["hol", "hour", "q"], rsuffix="_base")
        pts = pts.set_index("ts")
        arr = pts.loc[pts.index.intersection(imp.index[imp.event_arrive == 1])]
        dep = pts.loc[pts.index.intersection(imp.index[imp.event_depart == 1])]
        sh = int(pd.to_datetime(g.start_time, format="%H:%M").dt.hour.mode().iloc[0])
        eh = int(pd.to_datetime(g.end_time, format="%H:%M").dt.hour.mode().iloc[0])
        day_e = e.groupby("date")[["entries", "entries_base"]].sum()
        rows.append({
            "venue": venue, "station": st, "n_event_days": e.date.nunique(),
            "typical_start": f"{sh}:00", "typical_end": f"{eh}:00",
            "arrival_exits_ratio": arr.exits.sum() / arr.exits_base.sum(),
            "arrival_extra_exits_per_event": (arr.exits - arr.exits_base).sum() / len(g),
            "departure_entries_ratio": dep.entries.sum() / dep.entries_base.sum(),
            "departure_extra_entries_per_event": (dep.entries - dep.entries_base).sum() / len(g),
            "daily_entries_ratio": day_e.entries.sum() / day_e.entries_base.sum(),
        })
        curves[venue] = (st, e.groupby("hour")[["entries", "entries_base", "exits", "exits_base"]].mean())
        # data-driven gap check: non-event days whose evening (20-23h) entries at the venue station are
        # >= 2x the baseline -> candidate unlisted events (NOT added to events.csv, for manual review)
        ev_days = set(pd.to_datetime(esh.loc[esh.station == st, "ts"]).dt.normalize())
        nb = pts[(pts.hour >= 20) & ~pts.date.isin(ev_days)]
        dd = nb.groupby("date")[["entries", "entries_base"]].sum()
        dd = dd[dd.entries_base > 0]
        dd = dd[dd.entries / dd.entries_base >= 2.0]
        for d, r in dd.iterrows():
            cands.append({"venue": venue, "station": st, "date": d.date(), "evening_entries": int(r.entries),
                          "baseline": round(r.entries_base), "ratio": r.entries / r.entries_base})
    pd.DataFrame(cands).to_csv(RESULTS / "eda_event_candidates_unlisted.csv", index=False, float_format="%.2f")
    NUM["unlisted_event_candidates"] = pd.DataFrame(cands).groupby("venue").size().to_dict() if cands else {}
    tab = pd.DataFrame(rows).sort_values("departure_entries_ratio", ascending=False)
    tab.to_csv(RESULTS / "eda_event_uplift.csv", index=False, float_format="%.2f")
    NUM["event_uplift"] = tab[["venue", "station", "n_event_days", "arrival_exits_ratio",
                               "departure_entries_ratio"]].round(2).to_dict("records")
    n = len(curves)
    fig, axes = plt.subplots((n + 1) // 2, 2, figsize=(10, 2.6 * ((n + 1) // 2)), squeeze=False)
    for ax, (venue, (st, c)) in zip(axes.flat, curves.items()):
        ax.plot(c.index, c.entries, color="#d62728", label="進站（活動日）")
        ax.plot(c.index, c.entries_base, color="#d62728", ls="--", alpha=0.6, label="進站（基準）")
        ax.plot(c.index, c.exits, color="#1f77b4", label="出站（活動日）")
        ax.plot(c.index, c.exits_base, color="#1f77b4", ls="--", alpha=0.6, label="出站（基準）")
        ax.set_title(f"{venue} → {st}站", fontsize=10)
        ax.set_xticks(range(0, 24, 3))
        ax.grid(alpha=0.3)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=7)
    fig.suptitle("活動日 vs 同日別同小時基準（同季、排除其他活動日）的周邊站人流", y=1.0)
    fig.tight_layout()
    fig.savefig(FIGURES / "eda_event_station_profiles.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- 4. typhoon
def typhoon_effect(net: pd.DataFrame, cal: pd.DataFrame):
    f = EXTERNAL / "typhoon_days.csv"
    if not f.exists():
        print("typhoon_days.csv missing -> skip")
        return
    ty = pd.read_csv(f, parse_dates=["date"])
    daily = net.groupby("date")[["entries"]].sum()
    daily["dow"] = daily.index.dayofweek
    daily = daily.join(cal.set_index("date")[["is_holiday"]])
    tyd = set(ty.date)
    rows = []
    for t in ty.itertuples(index=False):
        d = t.date
        if d not in daily.index:
            continue
        # baseline: same weekday, +-4 weeks, not holiday-type mismatch and not a typhoon day
        cand = [d + pd.Timedelta(weeks=k) for k in (-4, -3, -2, -1, 1, 2, 3, 4)]
        cand = [c for c in cand if c in daily.index and c not in tyd
                and daily.loc[c, "is_holiday"] == daily.loc[d, "is_holiday"]]
        base = daily.loc[cand, "entries"].mean()
        rows.append({"date": d.date(), "type": t.type, "note": t.note,
                     "entries": int(daily.loc[d, "entries"]), "baseline": round(base),
                     "ratio": daily.loc[d, "entries"] / base})
    tab = pd.DataFrame(rows)
    tab.to_csv(RESULTS / "eda_typhoon_days.csv", index=False, float_format="%.3f")
    NUM["typhoon_days"] = tab[["date", "type", "ratio"]].astype({"date": str}).round(3).to_dict("records")
    if len(tab):
        # hourly profile on typhoon days vs baseline
        fig, ax = plt.subplots(figsize=(7, 3.4))
        for r in tab.itertuples(index=False):
            d = pd.Timestamp(r.date)
            h = net[net.date == d].set_index("hour")["entries"]
            ax.plot(h.index, h / 1e3, lw=1.2, label=f"{r.date} {r.type}")
        wk = net[(net.hol == 0)].groupby("hour")["entries"].mean()
        ax.plot(wk.index, wk / 1e3, "k--", lw=1.5, label="平日平均")
        ax.set_xlabel("小時")
        ax.set_ylabel("全網進站（千人次）")
        ax.set_title("颱風停班停課日的全網每小時進站量")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        fig.savefig(FIGURES / "eda_typhoon_days.png", dpi=150)
        plt.close(fig)


def main():
    cal = load_calendar()
    panel = load_panel(hist=True)
    net = network_hourly(panel, cal)
    main_net = net[net.ts >= MAIN_START]
    fig_profile(net, main_net)

    w = load_weather("obs")
    if w is not None:
        tabs = []
        # typhoon work/school-suspension days are excluded (they confound heavy rain with closures);
        # a version including them is kept for comparison
        tyf = EXTERNAL / "typhoon_days.csv"
        ty = set(pd.to_datetime(pd.read_csv(tyf)["date"])) if tyf.exists() else set()
        nt = net[~net.date.isin(ty)]
        for period, d in [("2025-10~2026-09", nt[nt.ts >= MAIN_START]),
                          ("2025-10~2026-09 含颱風停班日（=進度報告定義）", main_net), ("2024-10~2026-09", nt),
                          ("2024-10~2026-09 含颱風停班日", net)]:
            t, df = rain_effect(d, w, period)
            tabs.append(t)
            if period == "2025-10~2026-09":
                fig_rain(t, "eda_rain_ratio.png", "降雨 × 日別：全網進站量相對無雨基準（2025/10–2026/9，排除颱風停班日，95% CI）")
                rain_by_hour_heat(df)
                NUM["rain_ratio_main"] = t[["day_type", "rain", "n_hours", "ratio_mean", "ci_low",
                                            "ci_high"]].round(3).to_dict("records")
        pd.concat(tabs).to_csv(RESULTS / "eda_rain_ratio.csv", index=False, float_format="%.3f")
        fig_rain(tabs[2], "eda_rain_ratio_2y.png", "降雨 × 日別：全網進站量相對無雨基準（2024/10–2026/9，排除颱風停班日，95% CI）")
        wf = load_weather("forecast")
        if wf is not None:
            m = w.merge(wf, on="ts", suffixes=("_obs", "_fc")).dropna(subset=["precip_obs", "precip_fc"])
            o, f = m.precip_obs >= 1, m.precip_fc >= 1
            fc = {"temp_mae": (m.temp_obs - m.temp_fc).abs().mean(),
                  "precip_mae": (m.precip_obs - m.precip_fc).abs().mean(),
                  "precip_corr": np.corrcoef(m.precip_obs, m.precip_fc)[0, 1],
                  "rain_hours_obs": int(o.sum()), "rain_hours_fc": int(f.sum()),
                  "rain_hit_rate": (o & f).sum() / max(o.sum(), 1),
                  "rain_false_alarm_ratio": (~o & f).sum() / max(f.sum(), 1)}
            pd.Series(fc).to_csv(RESULTS / "eda_forecast_vs_obs.csv", header=["value"], float_format="%.3f")
            NUM["forecast_vs_obs"] = {k: round(float(v), 3) for k, v in fc.items()}
    else:
        print("weather_hourly.csv missing -> skip rain EDA")

    event_effect(panel, cal)
    typhoon_effect(net, cal)
    (RESULTS / "eda_summary_numbers.json").write_text(json.dumps(NUM, ensure_ascii=False, indent=1, default=str))
    print(json.dumps(NUM, ensure_ascii=False, indent=1, default=str)[:3000])


if __name__ == "__main__":
    main()
