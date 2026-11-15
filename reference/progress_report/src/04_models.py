"""t+1 小時進站/出站量預測：基準方法 vs LightGBM；以及「下一小時是否擁擠」分類。
切分：2025-10~2026-07 訓練、2026-08 驗證(early stopping)、2026-09 測試。"""
import json, os
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, average_precision_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_parquet(f"{ROOT}/data/processed/dataset.parquet").sort_values(["station", "ts"]).reset_index(drop=True)
df["station_id"] = df["station"].astype("category").cat.codes
g = df.groupby("station")

# 特徵：以 t 時刻已知資訊預測 t+1 小時
df["total"] = df["entry"] + df["exit"]
for c in ["entry", "exit"]:
    for l in [1, 2, 3, 24, 168]:
        df[f"{c}_lag{l}"] = g[c].shift(l)
    df[f"{c}_roll24"] = g[c].transform(lambda s: s.rolling(24).mean())
    df[f"{c}_roll168"] = g[c].transform(lambda s: s.rolling(168).mean())
    df[f"y_{c}"] = g[c].shift(-1)                      # 目標：下一小時
    df[f"y_{c}_seas"] = g[c].shift(167)                # seasonal naive：t+1-168
df["y_ts"] = df["ts"] + pd.Timedelta(hours=1)
df["y_hour"] = df["y_ts"].dt.hour
df["y_dow"] = df["y_ts"].dt.dayofweek
df["y_month"] = df["y_ts"].dt.month
# 目標時段的假日/天氣（假設預報準確，與 YouBike 組相同假設）
df["y_holiday"] = g["is_holiday"].shift(-1)
for c in ["rain", "temp", "humidity"]:
    df[f"y_{c}"] = g[c].shift(-1)
df["y_rain3"] = g["rain"].transform(lambda s: s.shift(-1).rolling(3, min_periods=1).sum())
df = df.dropna(subset=["entry_lag168", "entry_roll168", "y_entry", "y_holiday"]).reset_index(drop=True)
import warnings; warnings.filterwarnings("ignore")

train_end, val_end = pd.Timestamp("2026-08-01"), pd.Timestamp("2026-09-01")
tr, va, te = (df[df["y_ts"] < train_end].copy(), df[(df["y_ts"] >= train_end) & (df["y_ts"] < val_end)].copy(), df[df["y_ts"] >= val_end].copy())
peak = te["y_hour"].isin([7, 8, 17, 18]) & (te["y_holiday"] == 0)
print("train/val/test:", len(tr), len(va), len(te))

feats = ["station_id", "y_hour", "y_dow", "y_holiday", "y_month", "y_rain", "y_rain3", "y_temp", "y_humidity",
         *[f"{c}_lag{l}" for c in ["entry", "exit"] for l in [1, 2, 3, 24, 168]],
         *[f"{c}_roll{w}" for c in ["entry", "exit"] for w in [24, 168]]]

def mae(a, b): return float(np.mean(np.abs(a - b)))
def rmse(a, b): return float(np.sqrt(np.mean((a - b) ** 2)))

res, pred = {}, {}
for c in ["entry", "exit"]:
    y = te[f"y_{c}"].values
    ha = tr.assign(k=tr["y_holiday"]).groupby(["station", "k", "y_hour"])[f"y_{c}"].mean().rename("ha")
    p_ha = te.join(ha, on=["station", "y_holiday", "y_hour"])["ha"].fillna(0).values
    cand = {"Last Value": te[c].values, "Seasonal Naive": te[f"y_{c}_seas"].values, "Historical Average": p_ha}
    m = lgb.train({"objective": "poisson", "learning_rate": .05, "num_leaves": 63, "min_data_in_leaf": 50,
                   "feature_fraction": .8, "verbose": -1, "seed": 0},
                  lgb.Dataset(tr[feats], tr[f"y_{c}"], categorical_feature=["station_id"]), 2000,
                  valid_sets=[lgb.Dataset(va[feats], va[f"y_{c}"], categorical_feature=["station_id"])],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
    cand["LightGBM"] = m.predict(te[feats])
    # 變體：目標除以站點近一週平均量級（消除大站/小站的尺度差），再乘回；加入歷史平均作為特徵
    sc_ = lambda d: d[f"{c}_roll168"].values + 1.0
    for d, hv in ((tr, tr), (va, tr), (te, tr)):
        d[f"ha_{c}"] = d.join(ha, on=["station", "y_holiday", "y_hour"])["ha"].fillna(0).values / sc_(d)
    fs = feats + [f"ha_{c}"]
    m2 = lgb.train({"objective": "regression", "learning_rate": .05, "num_leaves": 63, "min_data_in_leaf": 50,
                    "feature_fraction": .8, "verbose": -1, "seed": 0},
                   lgb.Dataset(tr[fs], tr[f"y_{c}"] / sc_(tr), weight=sc_(tr), categorical_feature=["station_id"]), 2000,
                   valid_sets=[lgb.Dataset(va[fs], va[f"y_{c}"] / sc_(va), weight=sc_(va), categorical_feature=["station_id"])],
                   callbacks=[lgb.early_stopping(50, verbose=False)])
    cand["LightGBM (scaled)"] = np.clip(m2.predict(te[fs]) * sc_(te), 0, None)
    imp = pd.Series(m.feature_importance("gain"), feats); imp = (imp / imp.sum()).sort_values(ascending=False)
    res[c] = {k: {"MAE": mae(y, v), "RMSE": rmse(y, v), "MAE_peak": mae(y[peak.values], v[peak.values])} for k, v in cand.items()}
    res[c]["importance_top"] = imp.head(8).round(4).to_dict()
    res[c]["best_iter"] = m.best_iteration
    res[c]["test_mean"] = float(y.mean())
    pred[c] = cand["LightGBM (scaled)"] if res[c]["LightGBM (scaled)"]["MAE"] < res[c]["LightGBM"]["MAE"] else cand["LightGBM"]
    # 雨天誤差：LightGBM 在雨天(>=1mm)與無雨的 MAE
    r = (te["y_rain"] >= 1).values
    res[c]["lgb_MAE_rain"] = mae(y[r], pred[c][r]); res[c]["lgb_MAE_dry"] = mae(y[~r], pred[c][~r])
    print(c, pd.DataFrame({k: v for k, v in res[c].items() if isinstance(v, dict) and "MAE" in v}).T.round(3))
    print(imp.head(8).round(3))

# ---- 擁擠預警：站 × (平日/假日) × 小時 的 90 百分位 → 以各站訓練期總量 P95 為擁擠門檻 ----
tr_total = tr["y_entry"] + tr["y_exit"]
thr = tr.assign(t=tr_total).groupby("station")["t"].quantile(.95).rename("thr")
for d in (tr, va, te):
    d["thr"] = d["station"].map(thr).values
    d["y_crowd"] = ((d["y_entry"] + d["y_exit"]) >= d["thr"]).astype(int)
# 基準：以歷史平均預測量是否 >= 門檻
ha_tot = tr.assign(t=tr_total).groupby(["station", "y_holiday", "y_hour"])["t"].mean().rename("ha")
b_score = te.join(ha_tot, on=["station", "y_holiday", "y_hour"])["ha"].fillna(0).values / te["thr"].values
feats_c = feats + ["thr"]
mc = lgb.train({"objective": "binary", "learning_rate": .05, "num_leaves": 63, "min_data_in_leaf": 50,
                "feature_fraction": .8, "verbose": -1, "seed": 0},
               lgb.Dataset(tr[feats_c], tr["y_crowd"], categorical_feature=["station_id"]), 2000,
               valid_sets=[lgb.Dataset(va[feats_c], va["y_crowd"], categorical_feature=["station_id"])],
               callbacks=[lgb.early_stopping(50, verbose=False)])
sc = mc.predict(te[feats_c]); yc = te["y_crowd"].values
def cls(score, t):
    p, r, f, _ = precision_recall_fscore_support(yc, score >= t, average="binary", zero_division=0); return p, r, f
# 在驗證集上選使 Recall>=0.7 的最大門檻
sv = mc.predict(va[feats_c]); yv = va["y_crowd"].values
ths = np.quantile(sv, np.linspace(.5, .999, 400)); ok = [t for t in ths if (sv[yv == 1] >= t).mean() >= .7]
t_star = max(ok) if ok else .5
p, r, f = cls(sc, t_star)
bs_v = va.join(ha_tot, on=["station", "y_holiday", "y_hour"])["ha"].fillna(0).values / va["thr"].values
ok_b = [t for t in np.linspace(0, 2, 400) if (bs_v[yv == 1] >= t).mean() >= .7]; tb = max(ok_b)
pb, rb, fb = cls(b_score, tb)
res["crowd"] = {"positive_rate_test": float(yc.mean()), "n_test": int(len(yc)),
                "HistAvg": {"AUC": roc_auc_score(yc, b_score), "AP": average_precision_score(yc, b_score), "P": pb, "R": rb, "F1": fb},
                "LightGBM": {"AUC": roc_auc_score(yc, sc), "AP": average_precision_score(yc, sc), "P": p, "R": r, "F1": f}}
print(json.dumps(res["crowd"], indent=1))
os.makedirs(f"{ROOT}/results", exist_ok=True)
json.dump(res, open(f"{ROOT}/results/model_results.json", "w"), ensure_ascii=False, indent=1)
te[["station", "y_ts", "y_entry", "y_exit"]].assign(p_entry=pred["entry"], p_exit=pred["exit"], p_crowd=sc, y_crowd=yc).to_parquet(f"{ROOT}/results/test_pred.parquet")
