"""Large events -> station x hour impact table.

Input : data/external/events.csv
        (name, venue, date, start_time, end_time, scale_estimate, source_url, confidence, stations)
        collected manually / by web search; every row carries a source URL.
Output: data/external/events_station_hour.csv
        ts, station, event_flag, event_scale, event_arrive, event_depart, n_events
        data/external/venue_station_map.csv   (the venue -> station mapping used)

Impact window (per event, per mapped station with weight w):
  arrival   : the 2 hours before start          -> event_arrive = 1   (people exit the station)
  departure : the 2 hours from the (estimated) end -> event_depart = 1  (people enter the station)
  long events (>= 6 h, e.g. exhibitions): every hour from start-1h to end+1h is flagged as well.
event_scale = sum over concurrent events of scale_estimate * w (persons), event_flag = 1 if any.
Station names follow data/processed/stations.csv.

Usage: python src/10_events.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import EXTERNAL, PROCESSED  # noqa: E402

# venue -> [(station, weight)]; weight ~ share of the crowd expected to use that station
VENUE_STATIONS: dict[str, list[tuple[str, float]]] = {
    "台北大巨蛋": [("國父紀念館", 1.0), ("市政府", 0.4), ("忠孝敦化", 0.3)],
    "台北小巨蛋": [("台北小巨蛋", 1.0), ("南京復興", 0.4)],
    "台北流行音樂中心": [("昆陽", 1.0), ("南港", 0.4)],
    "南港展覽館": [("南港展覽館", 1.0), ("南港軟體園區", 0.2)],
    "台北世貿": [("台北101/世貿", 1.0), ("市政府", 0.4), ("象山", 0.3)],
    "台北101跨年": [("台北101/世貿", 1.0), ("市政府", 1.0), ("象山", 0.8), ("國父紀念館", 0.6),
                  ("信義安和", 0.6), ("忠孝敦化", 0.4)],
    "大稻埕": [("北門", 1.0), ("大橋頭站", 1.0), ("雙連", 0.4), ("中山", 0.3)],
}
# Venues whose location varies (燈節, 路跑/遊行, 國慶 ...) carry an explicit `stations` column in
# events.csv, formatted "站名:權重;站名:權重" (e.g. "西門:1.0;北門:0.6"); it overrides the venue map.


def _ts(date: str, hm: str) -> pd.Timestamp:
    return pd.Timestamp(f"{date} {hm}")


def expand(ev: pd.DataFrame, stations: set[str]) -> pd.DataFrame:
    rows, unmapped = [], []
    for e in ev.itertuples(index=False):
        spec = getattr(e, "stations", None)
        if isinstance(spec, str) and spec.strip():
            st = [(x.split(":")[0].strip(), float(x.split(":")[1]) if ":" in x else 1.0)
                  for x in spec.split(";") if x.strip()]
            bad = [x for x, _ in st if x not in stations]
            assert not bad, f"unknown station(s) {bad} in event {e.name}"
        else:
            st = VENUE_STATIONS.get(e.venue, [])
        if not st:
            unmapped.append((e.name, e.venue))
            continue
        start = _ts(e.date, e.start_time).floor("h")
        end = _ts(e.date, e.end_time)
        if end <= _ts(e.date, e.start_time):
            end += pd.Timedelta(days=1)          # crosses midnight
        end = end.ceil("h") if end.minute else end
        scale = float(e.scale_estimate) if pd.notna(e.scale_estimate) else 0.0
        arrive = pd.date_range(start - pd.Timedelta(hours=2), start - pd.Timedelta(hours=1), freq="h")
        depart = pd.date_range(end, end + pd.Timedelta(hours=1), freq="h")
        during = (pd.date_range(start - pd.Timedelta(hours=1), end + pd.Timedelta(hours=1), freq="h")
                  if (end - start) >= pd.Timedelta(hours=6) else pd.DatetimeIndex([]))
        for s, w in st:
            for t in arrive:
                rows.append((t, s, scale * w, 1, 0))
            for t in depart:
                rows.append((t, s, scale * w, 0, 1))
            for t in during.difference(arrive).difference(depart):
                rows.append((t, s, scale * w, 0, 0))
    if unmapped:
        print(f"WARNING {len(unmapped)} events without station mapping, e.g. {unmapped[:5]}")
    df = pd.DataFrame(rows, columns=["ts", "station", "event_scale", "event_arrive", "event_depart"])
    df["n_events"] = 1
    out = df.groupby(["ts", "station"], as_index=False).agg(
        event_scale=("event_scale", "sum"), event_arrive=("event_arrive", "max"),
        event_depart=("event_depart", "max"), n_events=("n_events", "sum"))
    out["event_flag"] = 1
    out["event_scale"] = out["event_scale"].round(0)
    return out[["ts", "station", "event_flag", "event_scale", "event_arrive", "event_depart", "n_events"]]


def main():
    stations = set(pd.read_csv(PROCESSED / "stations.csv")["station"])
    for v, lst in VENUE_STATIONS.items():
        bad = [s for s, _ in lst if s not in stations]
        assert not bad, f"unknown station(s) {bad} for {v}"
    ev = pd.read_csv(EXTERNAL / "events.csv", dtype=str)
    ev["scale_estimate"] = pd.to_numeric(ev["scale_estimate"], errors="coerce")
    ev = ev.dropna(subset=["date", "start_time", "end_time"])
    out = expand(ev, stations)
    out = out[(out.ts >= "2024-10-01") & (out.ts < "2026-10-01")]
    out.to_csv(EXTERNAL / "events_station_hour.csv", index=False, date_format="%Y-%m-%d %H:%M:%S")
    pd.DataFrame([(v, s, w) for v, lst in VENUE_STATIONS.items() for s, w in lst],
                 columns=["venue", "station", "weight"]).to_csv(EXTERNAL / "venue_station_map.csv", index=False)
    print(f"{len(ev)} events ({ev.date.min()}..{ev.date.max()}), by venue:\n{ev.venue.value_counts().to_string()}")
    print(f"-> events_station_hour.csv {len(out)} rows, {out.station.nunique()} stations, "
          f"{out.ts.dt.normalize().nunique()} days")


if __name__ == "__main__":
    main()
