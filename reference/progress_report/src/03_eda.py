"""探索性分析：全網每小時進站量、天氣影響、站點排名。輸出 figures/ 與 results/eda.json"""
import json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "SimHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(f"{ROOT}/results", exist_ok=True)
df = pd.read_parquet(f"{ROOT}/data/processed/dataset.parquet")
df["hour"] = df["ts"].dt.hour
df["dow"] = df["ts"].dt.dayofweek
df["daytype"] = df["is_holiday"].map({0: "平日", 1: "假日"})

net = df.groupby(["ts", "hour", "daytype"], as_index=False)[["entry", "exit"]].sum()
prof = net.groupby(["daytype", "hour"])["entry"].mean().unstack(0)
fig, ax = plt.subplots(figsize=(6, 3))
prof.plot(ax=ax, marker="o", ms=3)
ax.set_xlabel("小時"); ax.set_ylabel("全網平均進站量（人次/小時）"); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(f"{ROOT}/figures/fig1_hourly_profile.png", dpi=200)

# 天氣：同日別、同小時下，雨天(>=1mm) vs 無雨(0mm) 的全網進站量比
net = net.merge(df.drop_duplicates("ts")[["ts", "rain", "temp"]], on="ts")
net["rain_grp"] = pd.cut(net["rain"], [-1, 0, 1, 5, 1e9], labels=["無雨", "<1mm", "1-5mm", ">5mm"])
base = net[net["rain_grp"] == "無雨"].groupby(["daytype", "hour"])["entry"].mean().rename("base")
net = net.join(base, on=["daytype", "hour"])
net["ratio"] = net["entry"] / net["base"]
rain = net[net["hour"].between(6, 23)].groupby(["daytype", "rain_grp"], observed=True)["ratio"].agg(["mean", "count"]).round(3)
print(rain)

# 站點排名
st = df.groupby("station")[["entry", "exit"]].sum()
st["total"] = st.sum(axis=1)
top = (st["total"].sort_values(ascending=False).head(10) / df["ts"].nunique()).round(0)
print(top)
json.dump({"rain": rain.reset_index().astype({"rain_grp": str}).to_dict("records"),
           "top_station_avg_per_hour": top.to_dict(),
           "n_station": int(df["station"].nunique()),
           "avg_daily_entry": float(df["entry"].sum() / df["ts"].dt.normalize().nunique())},
          open(f"{ROOT}/results/eda.json", "w"), ensure_ascii=False, indent=1)
# 假日 / 平日日均
d = df.groupby(["ts"])["entry"].sum().groupby(lambda t: t.normalize()).sum()
hol = df.drop_duplicates("ts").set_index("ts")["is_holiday"].groupby(lambda t: t.normalize()).first()
print(d.groupby(hol).mean().round(0))
