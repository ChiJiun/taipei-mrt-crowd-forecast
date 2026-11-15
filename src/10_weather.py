"""Hourly Taipei weather: observations (reanalysis) and archived short-range forecasts.

Outputs (Asia/Taipei local time, ts = hour start, 'YYYY-MM-DD HH:00:00'):
  data/external/weather_hourly.csv           observed / reanalysis   (Open-Meteo archive API)
  data/external/weather_forecast_hourly.csv  archived forecast       (Open-Meteo historical-forecast API)

Columns: ts, temp (degC), precip (mm, mean over points), precip_max (mm, max over points),
         rh (%), wind (km/h), weather_code (WMO, Taipei-station point), n_points

Source notes
- CWA station data (466920) would be preferable, but the CWA open-data API needs an authorization key
  and CODiS is not reachable from the build environment, so we use Open-Meteo (no key):
  * archive-api      : ERA5 / ERA5-Land / ECMWF IFS reanalysis-analysis blend ("best_match")
  * historical-forecast-api : continuous series stitched from the first hours of each model run
    (Open-Meteo "best_match"; for Taiwan mostly JMA / ECMWF IFS / GFS).  Each hour therefore comes
    from a run issued roughly 0-6 h (at most ~12 h) before it.  We treat it as the forecast that would
    be available at the prediction origin t for horizons t+1/t+2 (see README section in eda_summary).
- Several points across Taipei / New Taipei are requested in ONE call per API to save quota
  (the free API is rate-limited per IP per day).  Raw JSON responses are cached under
  data/interim/weather_cache/ so reruns need no network.

Usage: python src/10_weather.py [--refresh] [--start 2024-10-01] [--end 2026-09-30]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import EXTERNAL, INTERIM  # noqa: E402

CACHE = INTERIM / "weather_cache"
CACHE.mkdir(parents=True, exist_ok=True)

# name, lat, lon  (first point = CWA Taipei station 466920, used for weather_code)
POINTS = [
    ("台北站466920", 25.0377, 121.5149),
    ("松山/內湖", 25.0631, 121.5755),
    ("士林/北投", 25.1170, 121.5100),
    ("板橋", 25.0127, 121.4627),
    ("信義/南港", 25.0330, 121.6000),
]
VARS = ["temperature_2m", "precipitation", "relative_humidity_2m", "wind_speed_10m", "weather_code"]
RENAME = {"temperature_2m": "temp", "precipitation": "precip", "relative_humidity_2m": "rh",
          "wind_speed_10m": "wind", "weather_code": "weather_code"}

APIS = {
    "obs": "https://archive-api.open-meteo.com/v1/archive",
    "forecast": "https://historical-forecast-api.open-meteo.com/v1/forecast",
}
OUT = {"obs": EXTERNAL / "weather_hourly.csv", "forecast": EXTERNAL / "weather_forecast_hourly.csv"}


def fetch(kind: str, start: str, end: str, refresh: bool = False) -> list[dict]:
    cache = CACHE / f"{kind}_{start}_{end}.json"
    if cache.exists() and not refresh:
        return json.loads(cache.read_text())
    params = {
        "latitude": ",".join(f"{p[1]}" for p in POINTS),
        "longitude": ",".join(f"{p[2]}" for p in POINTS),
        "start_date": start, "end_date": end,
        "hourly": ",".join(VARS), "timezone": "Asia/Taipei",
    }
    url = APIS[kind] + "?" + urllib.parse.urlencode(params)
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=180) as r:
                data = json.loads(r.read())
            break
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")
            if "limit exceeded" in body.lower():
                sys.exit(f"[{kind}] Open-Meteo quota exceeded: {body}\nRetry after the daily reset (00:00 UTC).")
            if attempt == 3:
                raise RuntimeError(f"[{kind}] HTTP {e.code}: {body}") from e
        except (urllib.error.URLError, TimeoutError):
            if attempt == 3:
                raise
        time.sleep(5 * 2 ** attempt)
    if isinstance(data, dict):
        if data.get("error"):
            sys.exit(f"[{kind}] Open-Meteo error: {data.get('reason')}")
        data = [data]
    cache.write_text(json.dumps(data))
    return data


def to_frame(data: list[dict]) -> pd.DataFrame:
    frames = []
    for i, loc in enumerate(data):
        h = pd.DataFrame(loc["hourly"]).rename(columns={"time": "ts", **RENAME})
        h["ts"] = pd.to_datetime(h["ts"])
        h["point"] = i
        frames.append(h)
    long = pd.concat(frames, ignore_index=True)
    g = long.groupby("ts")
    out = pd.DataFrame({
        "temp": g["temp"].mean(),
        "precip": g["precip"].mean(),
        "precip_max": g["precip"].max(),
        "rh": g["rh"].mean(),
        "wind": g["wind"].mean(),
        "n_points": g["temp"].count(),
    })
    wc = long[long["point"] == 0].set_index("ts")["weather_code"]
    out["weather_code"] = wc.reindex(out.index)
    out = out.reset_index()
    for c in ("temp", "rh", "wind"):
        out[c] = out[c].round(1)
    for c in ("precip", "precip_max"):
        out[c] = out[c].round(2)
    return out[["ts", "temp", "precip", "precip_max", "rh", "wind", "weather_code", "n_points"]]


def build(kind: str, start: str, end: str, refresh: bool) -> pd.DataFrame:
    df = to_frame(fetch(kind, start, end, refresh))
    full = pd.DataFrame({"ts": pd.date_range(f"{start} 00:00", f"{end} 23:00", freq="h")})
    df = full.merge(df, on="ts", how="left")
    # short gaps (<=3h) are interpolated; precipitation gaps filled with 0 only if isolated
    num = ["temp", "rh", "wind"]
    df[num] = df[num].interpolate(limit=3, limit_area="inside")
    df[["precip", "precip_max"]] = df[["precip", "precip_max"]].fillna(0) \
        .where(df["precip"].notna() | df["temp"].notna())
    df["weather_code"] = df["weather_code"].ffill(limit=3)
    df.to_csv(OUT[kind], index=False, date_format="%Y-%m-%d %H:%M:%S")
    miss = df["temp"].isna().mean()
    rain_h = (df["precip"] >= 1).sum()
    print(f"[{kind}] {len(df)} hours {df.ts.min()} .. {df.ts.max()}  missing temp {miss:.2%}  "
          f"hours precip>=1mm {rain_h}  -> {OUT[kind].relative_to(OUT[kind].parents[2])}")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2024-10-01")
    ap.add_argument("--end", default="2026-09-30")
    ap.add_argument("--refresh", action="store_true", help="ignore cached responses")
    ap.add_argument("--kinds", default="obs,forecast")
    a = ap.parse_args()
    res = {}
    for k in a.kinds.split(","):
        res[k] = build(k, a.start, a.end, a.refresh)
    if {"obs", "forecast"} <= res.keys():
        m = res["obs"].merge(res["forecast"], on="ts", suffixes=("_obs", "_fc")).dropna()
        mae_t = (m.temp_obs - m.temp_fc).abs().mean()
        mae_p = (m.precip_obs - m.precip_fc).abs().mean()
        r_obs, r_fc = m.precip_obs >= 1, m.precip_fc >= 1
        hit = (r_obs & r_fc).sum() / max(r_obs.sum(), 1)
        far = (~r_obs & r_fc).sum() / max(r_fc.sum(), 1)
        print(f"forecast vs obs: temp MAE {mae_t:.2f} C, precip MAE {mae_p:.3f} mm, "
              f"rain(>=1mm) hit rate {hit:.2f}, false-alarm ratio {far:.2f}, corr precip "
              f"{np.corrcoef(m.precip_obs, m.precip_fc)[0, 1]:.2f}")


if __name__ == "__main__":
    main()
