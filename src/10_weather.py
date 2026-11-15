"""Hourly Taipei weather: observations (reanalysis) and archived short-range forecasts.

Outputs (Asia/Taipei local time, ts = hour start, 'YYYY-MM-DD HH:00:00'), columns ts, temp, precip, rh:
  data/external/weather_hourly.csv           observed / reanalysis, 2024-10-01 .. 2026-09-30
  data/external/weather_forecast_hourly.csv  archived forecast,     2024-10-01 .. 2026-09-30

Sources (Open-Meteo, no key; one grid point lat 25.0615 lon 121.5194 = same as the main-period file
supplied on main, so main-period values stay identical):
  * archive-api.open-meteo.com  (ERA5 / ERA5-Land / IFS "best_match")
      2025-10..2026-09 : data/external/openmeteo_archive_raw_2025-10_2026-09.json   (from main, not refetched)
      2024-10..2025-09 : data/external/openmeteo_archive_raw_2024-10_2025-09.json   (fetched here)
  * historical-forecast-api.open-meteo.com
      data/external/openmeteo_histforecast_raw_2024-10_2026-09.json
    The historical-forecast series is stitched from the first hours of successive model runs
    (best_match; for Taiwan mainly JMA MSM/GSM, ECMWF IFS, GFS), so each hour comes from a run issued
    roughly 1-6 h earlier (runs every 1-6 h, plus ~1-3 h publication delay). We therefore treat it as
    the forecast available at the prediction origin t for targets t+1 / t+2. It is NOT a strict
    "issued before t" archive: for a t+2 target the true lead may be slightly shorter than assumed.
Raw JSON responses are committed so reruns need no network (the free API has a per-IP daily quota);
--refresh forces a new download.

The CWA (中央氣象署) station 466920 would be preferable but its open-data API needs an authorization
key (not available) and CODiS is not reachable from the build environment.

Usage: python src/10_weather.py [--refresh] [--kinds obs,forecast]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import EXTERNAL  # noqa: E402

LAT, LON = 25.0615, 121.5194
VARS = ["temperature_2m", "precipitation", "relative_humidity_2m"]
RENAME = {"time": "ts", "temperature_2m": "temp", "precipitation": "precip", "relative_humidity_2m": "rh"}
START, END = "2024-10-01", "2026-09-30"

ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
HISTFC = "https://historical-forecast-api.open-meteo.com/v1/forecast"
RAW = {
    "obs_main": (EXTERNAL / "openmeteo_archive_raw_2025-10_2026-09.json", ARCHIVE, "2025-10-01", "2026-09-30"),
    "obs_hist": (EXTERNAL / "openmeteo_archive_raw_2024-10_2025-09.json", ARCHIVE, "2024-10-01", "2025-09-30"),
    "forecast": (EXTERNAL / "openmeteo_histforecast_raw_2024-10_2026-09.json", HISTFC, START, END),
}
OUT = {"obs": EXTERNAL / "weather_hourly.csv", "forecast": EXTERNAL / "weather_forecast_hourly.csv"}


def fetch(key: str, refresh: bool = False) -> dict:
    path, api, s, e = RAW[key]
    if path.exists() and not (refresh and key != "obs_main"):
        return json.loads(path.read_text())
    params = {"latitude": LAT, "longitude": LON, "start_date": s, "end_date": e,
              "hourly": ",".join(VARS), "timezone": "Asia/Taipei"}
    url = api + "?" + urllib.parse.urlencode(params)
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=180) as r:
                data = json.loads(r.read())
            break
        except urllib.error.HTTPError as err:
            body = err.read().decode(errors="replace")
            if "limit exceeded" in body.lower():
                sys.exit(f"[{key}] Open-Meteo daily quota exceeded; retry after 00:00 UTC.\n{body}")
            if attempt == 3:
                raise RuntimeError(f"[{key}] HTTP {err.code}: {body}") from err
        except (urllib.error.URLError, TimeoutError):
            if attempt == 3:
                raise
        time.sleep(5 * 2 ** attempt)
    if data.get("error"):
        sys.exit(f"[{key}] Open-Meteo error: {data.get('reason')}")
    path.write_text(json.dumps(data))
    print(f"[{key}] downloaded -> {path.name}")
    return data


def to_frame(data: dict) -> pd.DataFrame:
    h = pd.DataFrame(data["hourly"]).rename(columns=RENAME)
    h["ts"] = pd.to_datetime(h["ts"])
    return h[["ts", "temp", "precip", "rh"]]


def finalize(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    full = pd.DataFrame({"ts": pd.date_range(f"{START} 00:00", f"{END} 23:00", freq="h")})
    df = full.merge(df.drop_duplicates("ts"), on="ts", how="left")
    miss = df[["temp", "precip", "rh"]].isna().mean()
    df.to_csv(OUT[kind], index=False, date_format="%Y-%m-%d %H:%M:%S")
    print(f"[{kind}] {len(df)} hours {df.ts.min()} .. {df.ts.max()}; missing "
          f"{', '.join(f'{c} {v:.2%}' for c, v in miss.items())}; hours precip>=1mm "
          f"{int((df.precip >= 1).sum())} -> {OUT[kind].name}")
    return df


def build_obs(refresh: bool) -> pd.DataFrame:
    main = to_frame(fetch("obs_main"))
    hist = to_frame(fetch("obs_hist", refresh))
    hist = hist[hist.ts < main.ts.min()]
    df = finalize(pd.concat([hist, main], ignore_index=True).sort_values("ts"), "obs")
    # main-period values must equal the file supplied on main (built from the same raw json)
    m = df[df.ts >= main.ts.min()].reset_index(drop=True)
    assert np.allclose(m[["temp", "precip", "rh"]].to_numpy(dtype=float),
                       main[["temp", "precip", "rh"]].to_numpy(dtype=float), equal_nan=True)
    return df


def build_forecast(refresh: bool) -> pd.DataFrame:
    return finalize(to_frame(fetch("forecast", refresh)), "forecast")


def compare(obs: pd.DataFrame, fc: pd.DataFrame):
    m = obs.merge(fc, on="ts", suffixes=("_obs", "_fc")).dropna()
    o, f = m.precip_obs >= 1, m.precip_fc >= 1
    print(f"forecast vs obs ({len(m)} h): temp MAE {(m.temp_obs - m.temp_fc).abs().mean():.2f} C, "
          f"precip MAE {(m.precip_obs - m.precip_fc).abs().mean():.3f} mm, corr "
          f"{np.corrcoef(m.precip_obs, m.precip_fc)[0, 1]:.2f}; rain>=1mm: obs {o.sum()} h, fc {f.sum()} h, "
          f"hit {(o & f).sum() / max(o.sum(), 1):.2f}, false alarm {(~o & f).sum() / max(f.sum(), 1):.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="re-download (except the main-period file)")
    ap.add_argument("--kinds", default="obs,forecast")
    a = ap.parse_args()
    res = {}
    for k in a.kinds.split(","):
        res[k] = build_obs(a.refresh) if k == "obs" else build_forecast(a.refresh)
    if len(res) == 2:
        compare(res["obs"], res["forecast"])


if __name__ == "__main__":
    main()
