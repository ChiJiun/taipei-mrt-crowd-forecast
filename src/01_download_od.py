"""Download monthly Taipei Metro hourly OD files and aggregate on the fly.

For each month YYYYMM the raw ~300MB csv (日期,時段,進站,出站,人次) is streamed to
data/raw, reduced to three small tables, then deleted:

  interim/station_hour_YYYYMM.parquet  date, hour, station, entries, exits
  interim/od_daily_YYYYMM.parquet      date, origin, dest, trips   (for spatial / OD-flow features)
  interim/od_hourprofile_YYYYMM.parquet hour, origin, dest, trips  (month sum by hour)

Usage:  python src/01_download_od.py 2025-10 2026-09   [--workers 3]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import EXTERNAL, INTERIM, RAW  # noqa: E402

LIST_URL = ("https://data.taipei/api/dataset/63f31c7e-7fc3-418b-bd82-b95158755b4d/"
            "resource/eb481f58-1238-4cff-8caa-fa7bb20cb4f4/download")


def file_list() -> pd.DataFrame:
    path = EXTERNAL / "od_file_list.csv"
    if not path.exists():
        subprocess.run(["curl", "-sSfL", "-m", "60", "-o", str(path), LIST_URL], check=True)
    df = pd.read_csv(path)
    df["ym"] = df["西元年"].astype(int).astype(str) + "-" + df["月"].astype(int).map("{:02d}".format)
    # blob host only answers through the proxy over https
    df["URL"] = df["URL"].str.strip().str.replace("http://", "https://", n=1, regex=False)
    return df[["ym", "URL"]]


def process_month(ym: str, url: str) -> str:
    tag = ym.replace("-", "")
    out = INTERIM / f"station_hour_{tag}.parquet"
    if out.exists():
        return f"{ym}: cached"
    raw = RAW / f"od_{tag}.csv"
    for attempt in range(4):
        r = subprocess.run(["curl", "-sSfL", "--retry", "3", "-m", "1800", "-o", str(raw), url])
        if r.returncode == 0:
            break
    else:
        raise RuntimeError(f"download failed {ym}")

    ent, ext, od_d, od_h = [], [], [], []
    reader = pd.read_csv(raw, chunksize=3_000_000, encoding="utf-8-sig",
                         dtype={"日期": "string", "時段": "int8", "進站": "category",
                                "出站": "category", "人次": "int32"})
    for ch in reader:
        ch.columns = ["date", "hour", "origin", "dest", "trips"]
        ch = ch[ch["trips"] > 0]
        ent.append(ch.groupby(["date", "hour", "origin"], observed=True)["trips"].sum())
        ext.append(ch.groupby(["date", "hour", "dest"], observed=True)["trips"].sum())
        od_d.append(ch.groupby(["date", "origin", "dest"], observed=True)["trips"].sum())
        od_h.append(ch.groupby(["hour", "origin", "dest"], observed=True)["trips"].sum())

    def _cat(parts, keys):
        s = pd.concat(parts)
        s = s.groupby(level=list(range(s.index.nlevels)), observed=True).sum()
        s.index.names = keys
        return s

    e = _cat(ent, ["date", "hour", "station"]).rename("entries")
    x = _cat(ext, ["date", "hour", "station"]).rename("exits")
    sh = pd.concat([e, x], axis=1).fillna(0).astype("int32").reset_index()
    sh["station"] = sh["station"].astype(str)
    sh.to_parquet(out, index=False)

    _cat(od_d, ["date", "origin", "dest"]).rename("trips").reset_index().astype(
        {"origin": str, "dest": str}).to_parquet(INTERIM / f"od_daily_{tag}.parquet", index=False)
    _cat(od_h, ["hour", "origin", "dest"]).rename("trips").reset_index().astype(
        {"origin": str, "dest": str}).to_parquet(INTERIM / f"od_hourprofile_{tag}.parquet", index=False)
    raw.unlink(missing_ok=True)
    return f"{ym}: {len(sh):,} station-hours, {sh.entries.sum():,} entries"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("start")
    ap.add_argument("end")
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    fl = file_list()
    todo = fl[(fl.ym >= a.start) & (fl.ym <= a.end)]
    print(f"{len(todo)} months to process", flush=True)
    with ThreadPoolExecutor(a.workers) as ex:
        for msg in ex.map(lambda r: process_month(r.ym, r.URL), todo.itertuples()):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
