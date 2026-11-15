"""Build the complete station x hour panel from interim monthly aggregates.

Keeps only stations that have traffic in every month of the window (drops stations
opened/closed during the period), fills missing station-hours with 0.

Outputs
  processed/panel_station_hour.parquet       2025-10..2026-09 (main study)
  processed/panel_station_hour_hist.parquet  2024-10..2026-09 (extended backtest)
  processed/stations.csv                     station list + mean hourly volume
Columns: ts (datetime64, hour start), station (category), entries, exits (int32)
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import (HIST_START, INTERIM, MAIN_END, MAIN_START, PANEL_HIST_PATH,  # noqa: E402
                    PANEL_PATH, PROCESSED)

# A few stations have been renamed across files; map to one canonical name.
RENAME = {"臺北車站": "台北車站", "臺大醫院": "台大醫院", "臺電大樓": "台電大樓",
          "臺北101/世貿": "台北101/世貿", "台北101世貿": "台北101/世貿",
          "臺北小巨蛋": "台北小巨蛋", "臺北橋": "台北橋"}


def load_months(start: str, end: str) -> pd.DataFrame:
    months = pd.period_range(start, end, freq="M")
    parts = []
    for m in months:
        f = INTERIM / f"station_hour_{m.strftime('%Y%m')}.parquet"
        if not f.exists():
            raise FileNotFoundError(f"{f} missing - run src/01_download_od.py first")
        d = pd.read_parquet(f)
        d["month"] = str(m)
        parts.append(d)
    df = pd.concat(parts, ignore_index=True)
    df["station"] = df["station"].str.strip().replace(RENAME)
    df = df.groupby(["date", "hour", "station", "month"], as_index=False)[["entries", "exits"]].sum()
    df["ts"] = pd.to_datetime(df["date"]) + pd.to_timedelta(df["hour"].astype(int), unit="h")
    return df


def build(start: str, end: str, out: Path) -> pd.DataFrame:
    df = load_months(start, end)
    n_months = df["month"].nunique()
    act = df[df.entries + df.exits > 0].groupby("station")["month"].nunique()
    keep = sorted(act[act == n_months].index)
    dropped = sorted(set(df.station) - set(keep))
    print(f"[{start}..{end}] stations kept {len(keep)}, dropped {len(dropped)}: {dropped}")

    idx = pd.MultiIndex.from_product(
        [pd.date_range(f"{start}-01", pd.Period(end).end_time.floor("h"), freq="h"), keep],
        names=["ts", "station"])
    panel = (df[df.station.isin(keep)].set_index(["ts", "station"])[["entries", "exits"]]
             .reindex(idx, fill_value=0).astype("int32").reset_index())
    panel["station"] = pd.Categorical(panel["station"], categories=keep)
    panel.to_parquet(out, index=False)
    print(f"  -> {out.name}: {len(panel):,} rows, {panel.ts.nunique()} hours")
    return panel


def main():
    panel = build(MAIN_START, MAIN_END, PANEL_PATH)
    st = (panel.assign(total=panel.entries + panel.exits).groupby("station", observed=True)
          .agg(mean_entries=("entries", "mean"), mean_exits=("exits", "mean"), mean_total=("total", "mean"))
          .sort_values("mean_total", ascending=False).round(1))
    st.to_csv(PROCESSED / "stations.csv")
    print(st.head(5))
    try:
        build(HIST_START, MAIN_END, PANEL_HIST_PATH)
    except FileNotFoundError as e:
        print("skip hist panel:", e)


if __name__ == "__main__":
    main()
