"""下載臺北捷運「每日分時各站OD流量」月檔，聚合成 站×小時 的進站/出站量後存檔。
原始檔每月約 300MB，聚合後刪除。連結清單來自 data.gov.tw/dataset/128506。"""
import sys, os, urllib.request
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW, OUT = f"{ROOT}/data/raw", f"{ROOT}/data/processed"
links = pd.read_csv(f"{RAW}/od_links.csv")
start, end = (2025, 10), (2026, 9)

for _, r in links.iterrows():
    ym = (r["西元年"], r["月"])
    if not (start <= ym <= end):
        continue
    tag = f"{ym[0]}{ym[1]:02d}"
    out = f"{OUT}/hourly_{tag}.parquet"
    if os.path.exists(out):
        continue
    tmp = f"{RAW}/od_{tag}.csv"
    print("下載", tag, flush=True)
    urllib.request.urlretrieve(r["URL"], tmp)
    df = pd.read_csv(tmp, encoding="utf-8-sig", dtype={"時段": int})
    df.columns = ["date", "hour", "origin", "dest", "n"]
    ent = df.groupby(["date", "hour", "origin"], as_index=False)["n"].sum().rename(columns={"origin": "station", "n": "entry"})
    ext = df.groupby(["date", "hour", "dest"], as_index=False)["n"].sum().rename(columns={"dest": "station", "n": "exit"})
    agg = ent.merge(ext, on=["date", "hour", "station"], how="outer").fillna(0)
    agg.to_parquet(out)
    os.remove(tmp)
    print("  完成", tag, len(agg), "列", flush=True)
