"""合併 站×小時 進出量、天氣、行事曆，產生建模用資料表 data/processed/dataset.parquet"""
import glob, json, os
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P, RAW = f"{ROOT}/data/processed", f"{ROOT}/data/raw"

df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{P}/hourly_*.parquet"))], ignore_index=True)
df["ts"] = pd.to_datetime(df["date"]) + pd.to_timedelta(df["hour"], unit="h")
df = df.groupby(["station", "ts"], as_index=False)[["entry", "exit"]].sum()

# 只保留資料期間內持續有運量的站（排除期間內新開通／停用的站）
span = df.groupby("station")["ts"].agg(["min", "max"])
full = span["min"].eq(df["ts"].min()) & span["max"].eq(df["ts"].max())
df = df[df["station"].isin(span.index[full])]

# 補齊 站 × 每小時 完整網格（沒有紀錄的時段視為 0）
idx = pd.MultiIndex.from_product([sorted(df["station"].unique()), pd.date_range(df["ts"].min(), df["ts"].max(), freq="h")], names=["station", "ts"])
df = df.set_index(["station", "ts"]).reindex(idx, fill_value=0).reset_index()

# 天氣
w = json.load(open(f"{RAW}/weather.json"))["hourly"]
w = pd.DataFrame({"ts": pd.to_datetime(w["time"]), "temp": w["temperature_2m"], "rain": w["precipitation"], "humidity": w["relative_humidity_2m"]})
df = df.merge(w, on="ts", how="left")

# 行事曆
cal = pd.concat([pd.DataFrame(json.load(open(f))) for f in glob.glob(f"{RAW}/cal_*.json")])
cal["day"] = pd.to_datetime(cal["date"], format="%Y%m%d")
cal["is_holiday"] = cal["isHoliday"].astype(int)
df["day"] = df["ts"].dt.normalize()
df = df.merge(cal[["day", "is_holiday"]], on="day", how="left")
df["is_holiday"] = df["is_holiday"].fillna(0).astype(int)

df = df.drop(columns="day").sort_values(["station", "ts"]).reset_index(drop=True)
df.to_parquet(f"{P}/dataset.parquet")
print(df.shape, df["station"].nunique(), "站", df["ts"].min(), "~", df["ts"].max())
print(df.isna().sum()[lambda s: s > 0])
