"""Workstream C-1: classical time-series baselines (SARIMAX, Prophet) vs Historical Average / LightGBM.

Same split as everywhere (config.py): parameters estimated on data <= TRAIN_END, validation = 2026-08,
test = 2026-09. Rolling t+1 / t+2 forecasts on the test month, aligned row-by-row with
results/pred_test_h{h}_{target}.parquet (produced by src/04_baselines_t1.py).

Models (one per station x target, all on sqrt(y) to stabilise variance, back-transformed by squaring):
  * SARIMAX  : regression with SARIMA(2,0,0)(1,0,0)_24 errors. Exog = historical-average profile
               (station x holiday x hour mean of sqrt(y), training only), weekly Fourier terms
               (period 168, K=3) and, if data/external/weather_hourly.csv exists, precipitation.
               Parameters are fitted once on the last FIT_HOURS hours of training; the Kalman filter is
               then run over validation+test with fixed parameters (= results.append/extend without
               re-estimation), so each forecast only uses data up to its origin t.
               t+1 = one-step prediction; t+2 = Z (T a_{t+1|t} + c) + d_{t+2} from the filtered state.
  * Prophet  : additive Prophet with weekly seasonality, separate daily seasonality for workdays /
               holidays (conditional seasonalities), holiday regressor (+ precipitation if available).
               Re-fitted every 7 days on the previous PROPHET_WEEKS weeks (rolling origin, weekly), and
               forecasts the next 7 days -> pure curve forecast, does NOT use the latest hour.
  * Prophet+AR: Prophet forecast + AR(1) correction of the latest residual, phi^h * e_t
               (phi estimated per station on the in-sample residuals of each fit). This makes Prophet a
               genuine t+h rolling forecast.

Usage:
  python src/30_timeseries.py                 # all 122 stations, h=1,2, entries+exits (~25 min, 4 cores)
  python src/30_timeseries.py --quick         # representative stations only (2 per cluster + 台北車站)
  python src/30_timeseries.py --methods sarimax --horizons 1
Outputs: results/ts_compare.csv/.md, results/ts_station_metrics.csv,
         results/pred_ts_test_h{h}_{target}.parquet (gitignored), figures/ts_example.png
         (--quick writes *_quick files and leaves the full-run outputs untouched)
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from config import FIGURES, RESULTS, TEST_END, TEST_START, TRAIN_END  # noqa: E402
from features import load_calendar, load_panel, load_weather  # noqa: E402
from metrics import reg_metrics  # noqa: E402

FIT_HOURS = 12 * 7 * 24       # SARIMAX estimation window (last 12 weeks of training)
PROPHET_WEEKS = 12            # Prophet training window per weekly refit
ORDER, SORDER = (2, 0, 0), (1, 0, 0, 24)
FOURIER_K = 3
QUICK_EXTRA = ["台北車站"]


# ----------------------------------------------------------------------------- data helpers
def series_frame(panel: pd.DataFrame, station: str, target: str, cal: pd.DataFrame,
                 weather: pd.DataFrame | None) -> pd.DataFrame:
    s = panel[panel.station == station][["ts", target]].rename(columns={target: "y"}).reset_index(drop=True)
    s["y"] = s["y"].astype(float)
    s["hour"] = s.ts.dt.hour
    s["hol"] = s.ts.dt.normalize().map(cal["is_holiday"]).fillna(False).astype(int)
    wk = (s.ts.dt.dayofweek * 24 + s.ts.dt.hour).to_numpy()
    for k in range(1, FOURIER_K + 1):
        s[f"w_sin{k}"] = np.sin(2 * np.pi * k * wk / 168)
        s[f"w_cos{k}"] = np.cos(2 * np.pi * k * wk / 168)
    if weather is not None and "precip" in weather:
        s = s.merge(weather[["ts", "precip"]], on="ts", how="left")
        s["precip"] = s["precip"].fillna(0.0)
    return s


def sarimax_exog(s: pd.DataFrame, z: np.ndarray, tr: np.ndarray) -> np.ndarray:
    ha = pd.Series(z[tr]).groupby([s.hol.values[tr], s.hour.values[tr]]).mean()
    HA = ha.reindex(pd.MultiIndex.from_arrays([s.hol, s.hour])).to_numpy()
    cols = [HA] + [s[c].to_numpy() for c in s.columns if c.startswith("w_")]
    if "precip" in s:
        cols.append(np.log1p(s["precip"].to_numpy()))
    return np.column_stack(cols)


# ----------------------------------------------------------------------------- SARIMAX
def run_sarimax(s: pd.DataFrame, horizons) -> tuple[dict, float, float]:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    z = np.sqrt(s.y.to_numpy())
    tr = (s.ts <= TRAIN_END).to_numpy()
    X = sarimax_exog(s, z, tr)
    idx = np.where(tr)[0][-FIT_HOURS:]
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mod = SARIMAX(z[idx], exog=X[idx], order=ORDER, seasonal_order=SORDER, trend="c")
        res = mod.fit(disp=False, maxiter=200)
    fit_s = time.time() - t0
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        full = res.apply(z, exog=X)            # fixed parameters, filter over the whole sample
    fr = full.filter_results
    out = {}
    p1 = fr.forecasts[0]                         # one-step predictions: value at t uses data <= t-1
    out[1] = p1
    if 2 in horizons:
        a = fr.predicted_state[:, : len(z)]      # a_{t|t-1}
        T = full.model["transition"]
        Z = full.model["design"]
        c = full.model["state_intercept"]
        d = full.model["obs_intercept"]
        c = c[:, :1] if c.ndim == 2 and c.shape[1] > 1 else c.reshape(-1, 1)
        a2 = T @ a + c                           # column t: a_{t+1|t-1}
        p2 = np.full(len(z), np.nan)
        dd = d[0] if d.ndim == 2 else np.full(len(z), float(d))
        p2[1:] = (Z @ a2)[0, :-1] + dd[1:]       # value at t+1 from origin t-1
        out[2] = p2
    pred_s = time.time() - t0
    return {h: np.clip(v, 0, None) ** 2 for h, v in out.items() if h in horizons}, fit_s, pred_s


# ----------------------------------------------------------------------------- Prophet
def run_prophet(s: pd.DataFrame, horizons) -> tuple[dict, float, float]:
    from prophet import Prophet
    logging.getLogger("prophet").setLevel(logging.ERROR)
    logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
    logging.getLogger("cmdstanpy").disabled = True
    z = np.sqrt(s.y.to_numpy())
    df = pd.DataFrame({"ds": s.ts, "y": z, "hol": s.hol.astype(float),
                       "is_hol": s.hol.astype(bool), "is_wd": ~s.hol.astype(bool)})
    if "precip" in s:
        df["rain"] = np.log1p(s["precip"].to_numpy())
    base = np.full(len(z), np.nan)
    corr = {h: np.full(len(z), np.nan) for h in horizons}
    fit_s = pred_s = 0.0
    origins = pd.date_range(TEST_START, TEST_END, freq="7D")       # 9/1, 9/8, 9/15, 9/22, 9/29
    for o in origins:
        trm = (df.ds < o) & (df.ds >= o - pd.Timedelta(weeks=PROPHET_WEEKS))
        fcm = (df.ds >= o) & (df.ds < o + pd.Timedelta(days=7))
        t0 = time.time()
        m = Prophet(daily_seasonality=False, weekly_seasonality=6, yearly_seasonality=False,
                    changepoint_prior_scale=0.01, uncertainty_samples=0)
        m.add_seasonality("daily_workday", period=1, fourier_order=12, condition_name="is_wd")
        m.add_seasonality("daily_holiday", period=1, fourier_order=12, condition_name="is_hol")
        m.add_regressor("hol")
        if "rain" in df:
            m.add_regressor("rain")
        m.fit(df[trm])
        fit_s += time.time() - t0
        t0 = time.time()
        # predict training window too (for residual AR) and the next week
        pm = trm | fcm
        yhat = m.predict(df[pm].drop(columns="y"))["yhat"].to_numpy()
        base_idx = np.where(pm)[0]
        fit_res = (df.y.to_numpy()[base_idx] - yhat)
        in_tr = trm.to_numpy()[base_idx]
        r = fit_res[in_tr]
        phi = float(np.clip(np.dot(r[1:], r[:-1]) / max(np.dot(r[:-1], r[:-1]), 1e-9), -0.99, 0.99))
        full_hat = np.full(len(z), np.nan)
        full_hat[base_idx] = yhat
        fc_idx = np.where(fcm)[0]
        base[fc_idx] = yhat[~in_tr]
        e = z - full_hat                          # residuals known at origin (actuals are observed)
        for h in horizons:
            src = fc_idx - h                      # origin = target - h, residual at origin
            corr[h][fc_idx] = full_hat[fc_idx] + phi ** h * e[src]
        pred_s += time.time() - t0
    out = {("Prophet", h): np.clip(base, 0, None) ** 2 for h in horizons}
    out.update({("Prophet+AR", h): np.clip(corr[h], 0, None) ** 2 for h in horizons})
    return out, fit_s, pred_s


# ----------------------------------------------------------------------------- driver
_G: dict = {}


def _init(panel, cal, weather):
    _G.update(panel=panel, cal=cal, weather=weather)


def _work(args):
    station, target, methods, horizons = args
    s = series_frame(_G["panel"], station, target, _G["cal"], _G["weather"])
    te = ((s.ts >= TEST_START) & (s.ts <= TEST_END)).to_numpy()
    rows, timing = {}, []
    if s.y[s.ts <= TRAIN_END].sum() == 0:      # e.g. exits of an entry-only gate: constant zero
        for m in methods:
            names = ["SARIMAX"] if m == "sarimax" else ["Prophet", "Prophet+AR"]
            for n in names:
                for h in horizons:
                    rows[(n, h)] = np.zeros(te.sum())
                timing.append((n, 0.0, 0.0))
        return station, target, s.ts[te].to_numpy(), rows, timing
    if "sarimax" in methods:
        p, f, q = run_sarimax(s, horizons)
        for h, v in p.items():
            rows[("SARIMAX", h)] = v[te]
        timing.append(("SARIMAX", f, q))
    if "prophet" in methods:
        p, f, q = run_prophet(s, horizons)
        for k, v in p.items():
            rows[k] = v[te]
        timing += [("Prophet", f, q), ("Prophet+AR", f, q)]
    return station, target, s.ts[te].to_numpy(), rows, timing


def pick_stations(quick: bool, panel: pd.DataFrame) -> list[str]:
    allst = list(panel.station.cat.categories)
    if not quick:
        return allst
    f = RESULTS / "cluster_assignments.csv"
    if not f.exists():
        raise SystemExit("--quick needs results/cluster_assignments.csv (run src/31_clustering.py first)")
    a = pd.read_csv(f).sort_values("mean_entries", ascending=False)
    pick = []
    for _, g in a.groupby("cluster"):
        g = g[(g.mean_entries > 1) & (g.mean_exits > 1)] if (g.mean_exits > 1).any() else g
        pick += [g.station.iloc[0], g.station.iloc[len(g) // 2]]     # largest + median station
    return list(dict.fromkeys(QUICK_EXTRA + pick))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="representative stations only")
    ap.add_argument("--methods", nargs="+", default=["sarimax", "prophet"], choices=["sarimax", "prophet"])
    ap.add_argument("--horizons", nargs="+", type=int, default=[1, 2], choices=[1, 2])
    ap.add_argument("--targets", nargs="+", default=["entries", "exits"])
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    a = ap.parse_args()

    panel = load_panel()
    cal = load_calendar().set_index("date")
    weather = load_weather("obs")
    print("weather exog:", "precip" if weather is not None else "none (data/external/weather_hourly.csv missing)")
    stations = pick_stations(a.quick, panel)
    print(f"{len(stations)} stations x {len(a.targets)} targets, methods={a.methods}, horizons={a.horizons}")
    tasks = [(s, t, a.methods, a.horizons) for t in a.targets for s in stations]
    results = []
    t_all = time.time()
    with Pool(a.jobs, initializer=_init, initargs=(panel, cal, weather)) as pool:
        for i, r in enumerate(pool.imap_unordered(_work, tasks)):
            results.append(r)
            if (i + 1) % 20 == 0 or i + 1 == len(tasks):
                print(f"  {i + 1}/{len(tasks)} series done ({time.time() - t_all:.0f}s)", flush=True)

    timing = pd.DataFrame([(st, tg, m, f, q) for st, tg, _, _, tl in results for m, f, q in tl],
                          columns=["station", "target", "method", "fit_s", "pred_s"])
    out_rows, st_rows = [], []
    suffix = "_quick" if a.quick else ""
    for target in a.targets:
        for h in a.horizons:
            pf = RESULTS / f"pred_test_h{h}_{target}.parquet"
            if not pf.exists():
                print(f"!! {pf.name} missing: run `python src/04_baselines_t1.py --horizon {h}` for the comparison")
                continue
            base = pd.read_parquet(pf)
            base = base[base.station.isin(stations)].copy()
            base["station"] = base.station.astype(str)
            ts_parts = []
            for st, tg, tsv, rows, _ in results:
                if tg != target:
                    continue
                d = pd.DataFrame({"ts_target": tsv, "station": st})
                for (m, hh), v in rows.items():
                    if hh == h:
                        d[m] = v
                ts_parts.append(d)
            tsdf = pd.concat(ts_parts)
            ts_methods = [c for c in tsdf.columns if c not in ("ts_target", "station")]
            mrg = base.merge(tsdf, on=["ts_target", "station"], how="left")
            assert mrg[ts_methods].notna().all().all(), "missing TS predictions"
            keep = ["ts_target", "station", "y"] + ts_methods
            mrg[keep].to_parquet(RESULTS / f"pred_ts_test_h{h}_{target}{suffix}.parquet", index=False)
            methods = ["Seasonal Naive", "Historical Average"] + ts_methods + ["LightGBM", "LightGBM (scaled)"]
            for m in methods:
                tm = timing[(timing.method == m) & (timing.target == target)]
                out_rows.append({"target": target, "horizon": h, "method": m, "n_stations": len(stations),
                                 **reg_metrics(mrg.y, mrg[m], mrg),
                                 "fit_s_total": round(tm.fit_s.sum(), 1) if len(tm) else np.nan,
                                 "pred_s_total": round(tm.pred_s.sum(), 1) if len(tm) else np.nan})
                g = mrg.assign(ae=(mrg[m] - mrg.y).abs()).groupby("station")
                st_rows.append(pd.DataFrame({"target": target, "horizon": h, "method": m,
                                             "MAE": g.ae.mean(), "mean_y": g.y.mean()}).reset_index())
    cmp_ = pd.DataFrame(out_rows)
    cmp_.to_csv(RESULTS / f"ts_compare{suffix}.csv", index=False)
    pd.concat(st_rows).round(3).to_csv(RESULTS / f"ts_station_metrics{suffix}.csv", index=False)
    write_md(cmp_, stations, a, weather is not None, suffix)
    print(cmp_.to_string(index=False))
    if not a.quick:
        plot_example(stations)


def md_table(df: pd.DataFrame) -> str:
    """Minimal markdown table (avoids the optional `tabulate` dependency)."""
    def fmt(v):
        return "" if pd.isna(v) else (f"{v:,.2f}" if isinstance(v, (float, np.floating)) else str(v))
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join("---:" if pd.api.types.is_numeric_dtype(df[c]) else "---" for c in df.columns) + "|"
    body = ["| " + " | ".join(fmt(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, sep] + body)


def write_md(cmp_: pd.DataFrame, stations, a, has_rain: bool, suffix: str):
    lines = [f"# 時間序列對照（測試集 2026-09，{len(stations)} 站{'，代表站抽樣' if a.quick else '，全部站點'}）", "",
             "方法說明見 `src/30_timeseries.py` 檔頭。SARIMAX 參數只在訓練期最後 12 週估計一次，"
             "測試期以 Kalman filter 逐小時更新狀態（不重估參數）；Prophet 每 7 天以前 12 週資料重估，"
             "預測未來 7 天（不使用最新一小時資訊）；Prophet+AR 再以 AR(1) 修正最新殘差。"
             f"外生降雨：{'有（weather_hourly.csv 的 precip）' if has_rain else '無（尚未取得天氣檔）'}。", "",
             "耗時為所有站點×該目標的總秒數（單核計），LightGBM 等基準的訓練耗時見 `src/04_baselines_t1.py`（全部站點共用一個模型，約 1–2 分鐘）。", ""]
    show = ["method", "MAE", "RMSE", "peak_MAE"] + (["rain_MAE"] if "rain_MAE" in cmp_ else []) + ["fit_s_total", "pred_s_total"]
    for (tg, h), g in cmp_.groupby(["target", "horizon"], sort=False):
        lines += [f"## {'進站' if tg == 'entries' else '出站'} t+{h}", "", md_table(g[show]), ""]
    if a.quick:
        lines += ["抽樣站點：" + "、".join(stations), ""]
    (RESULTS / f"ts_compare{suffix}.md").write_text("\n".join(lines), encoding="utf-8")


def plot_example(stations):
    from plotting import plt
    f = RESULTS / "pred_ts_test_h1_entries.parquet"
    b = RESULTS / "pred_test_h1_entries.parquet"
    if not f.exists() or "台北車站" not in stations:
        return
    d = pd.read_parquet(f).merge(pd.read_parquet(b)[["ts_target", "station", "LightGBM (scaled)"]].assign(
        station=lambda x: x.station.astype(str)), on=["ts_target", "station"])
    d = d[(d.station == "台北車站") & (d.ts_target >= "2026-09-14") & (d.ts_target < "2026-09-21")]
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.plot(d.ts_target, d.y, color="black", lw=1.6, label="實際")
    for m, c in (("SARIMAX", "tab:blue"), ("Prophet", "tab:orange"), ("Prophet+AR", "tab:green"),
                 ("LightGBM (scaled)", "tab:red")):
        if m in d:
            ax.plot(d.ts_target, d[m], color=c, lw=1, alpha=0.85, label=m)
    ax.set_title("台北車站 t+1 進站量預測（2026/9/14–9/20）")
    ax.set_ylabel("人次／小時")
    ax.legend(ncol=5, fontsize=8)
    ax.grid(alpha=0.3)
    fig.savefig(FIGURES / "ts_example.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
