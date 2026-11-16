"""Workstream B task 2: observed vs forecast weather as features.

The main experiment uses the *observed* weather of the target hour, i.e. it assumes a perfect weather
forecast. Here we quantify how optimistic that is. For each target/horizon:
  (a) obs->obs   : train and test with observed weather (the optimistic number in reg_main)
  (b) obs->fc    : same model, but at test time feed the archived forecast (realistic deployment)
  (c) fc->fc     : train and test with forecast weather (model learns how much to trust the forecast)
  (d) no weather : lower bound
Rain at the origin hour (precip_origin) is always taken from observations — it is known at time t.
Overestimation = MAE(realistic) - MAE(obs->obs), also expressed as a share of the weather gain (d - a).

Needs data/external/weather_hourly.csv and weather_forecast_hourly.csv (workstream A).
Usage: python src/20_weather_forecast.py [--quick]
Uses JMA (weather_forecast_hourly.csv) as the main forecast and ECMWF IFS 0.25 (weather_forecast_hourly_ecmwf.csv)
as a sensitivity analysis.
Writes results/reg_weather_forecast.csv, results/reg_weather_forecast.md
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import EXTERNAL, RAIN_MM, RESULTS, TEST_END, TEST_START  # noqa: E402
from features import load_panel, load_weather  # noqa: E402
from metrics import reg_metrics  # noqa: E402

R = importlib.import_module("20_reglib")


def weather_skill(obs, fc):
    m = obs.merge(fc, on="ts", suffixes=("_obs", "_fc"))
    t = m[(m.ts >= TEST_START) & (m.ts <= TEST_END)]
    out = []
    for name, d in (("全期", m), ("測試月 2026-09", t)):
        ro, rf = d.precip_obs >= RAIN_MM, d.precip_fc >= RAIN_MM
        out.append({"period": name, "hours": len(d), "rain_hours_obs": int(ro.sum()), "rain_hours_fc": int(rf.sum()),
                    "precip_corr": round(d.precip_obs.corr(d.precip_fc), 3),
                    "precip_MAE_mm": round((d.precip_obs - d.precip_fc).abs().mean(), 3),
                    "rain_hit_rate(POD)": round((ro & rf).sum() / max(ro.sum(), 1), 3),
                    "false_alarm_ratio": round((rf & ~ro).sum() / max(rf.sum(), 1), 3),
                    "temp_MAE": round((d.temp_obs - d.temp_fc).abs().mean(), 2)})
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="entries h=1 only, learning_rate 0.1")
    a = ap.parse_args()
    obs = load_weather("obs")
    fcs = {"JMA": load_weather("forecast")}
    f_ec = EXTERNAL / "weather_forecast_hourly_ecmwf.csv"
    if f_ec.exists():
        fcs["ECMWF"] = pd.read_csv(f_ec, parse_dates=["ts"])
    if obs is None or fcs["JMA"] is None:
        sys.exit("needs both weather_hourly.csv and weather_forecast_hourly.csv")
    params = {"learning_rate": 0.1} if a.quick else {}
    panel = load_panel()
    combos = [("entries", 1)] if a.quick else [("entries", 1), ("exits", 1), ("entries", 2), ("exits", 2)]
    rows = []
    for tgt, h in combos:
        fo = R.build(panel, tgt, h, weather=obs)
        tro, vao, teo = R.split(fo)
        cols = R.cols_full(fo)
        wcols = [c for c in cols if c in obs.columns or c.startswith("precip") or c.startswith("rain_")]
        nw = [c for c in cols if c not in wcols]
        m_obs = R.fit(tro, vao, cols, R.B_OBJ, True, params=params)
        m_nw = R.fit(tro, vao, nw, R.B_OBJ, True, params=params)
        P = {"(a) 觀測→觀測（假設預報完全準確）": R.predict(m_obs, teo, cols)}
        for src, fc in fcs.items():
            ff = R.build(panel, tgt, h, weather=fc[[c for c in obs.columns if c in fc.columns]])
            ff["precip_origin"] = fo["precip_origin"].to_numpy()  # rain at origin t is observed
            trf, vaf, tef = R.split(ff)
            assert (teo.index == tef.index).all()
            m_fc = R.fit(trf, vaf, cols, R.B_OBJ, True, params=params)
            P[f"(b) 觀測訓練→{src} 預報測試（實際部署）"] = R.predict(m_obs, tef, cols)
            P[f"(c) {src} 預報→預報"] = R.predict(m_fc, tef, cols)
        P["(d) 不用天氣"] = R.predict(m_nw, teo, nw)
        rain = (teo["precip"] >= RAIN_MM).to_numpy()  # rain defined by OBSERVED precip for all scenarios
        for k, p in P.items():
            r = {"target": tgt, "horizon": h, "scenario": k, **reg_metrics(teo.y, p, teo),
                 "rain_bias": round(float((p - teo.y.to_numpy())[rain].mean()), 2)}
            rows.append(r)
        print(pd.DataFrame(rows).tail(len(P)).to_string(index=False), flush=True)
    df = pd.DataFrame(rows)
    for (tgt, h), g in df.groupby(["target", "horizon"]):
        ga, gd = g[g.scenario.str.startswith("(a)")].iloc[0], g[g.scenario.str.startswith("(d)")].iloc[0]
        for i in g.index:
            df.loc[i, "over_vs_a"] = round(df.loc[i, "MAE"] - ga.MAE, 2)
            df.loc[i, "rain_over_vs_a"] = round(df.loc[i, "rain_MAE"] - ga.rain_MAE, 2)
            gain = gd.rain_MAE - ga.rain_MAE
            df.loc[i, "rain_gain_kept_%"] = round(100 * (gd.rain_MAE - df.loc[i, "rain_MAE"]) / gain, 1) \
                if abs(gain) > 1e-9 else np.nan
    suffix = "_quick" if a.quick else ""
    df.to_csv(RESULTS / f"reg_weather_forecast{suffix}.csv", index=False)
    sk = pd.concat([weather_skill(obs, fc).assign(forecast=src) for src, fc in fcs.items()], ignore_index=True)
    sk = sk[["forecast"] + [c for c in sk.columns if c != "forecast"]]
    md = ["# 觀測天氣 vs 預報天氣（LightGBM-B，測試集 2026-09）\n",
          "「觀測」= Open-Meteo 再分析（ECMWF IFS 9 km 分析場拼接）；主要預報 = JMA historical forecast"
          "（與觀測代理值無關，約 0–6 小時前發布）；ECMWF IFS 0.25° 為敏感度分析（與觀測同模式家族，誤差偏低，偏樂觀）。"
          "預測起點 t 的降雨（precip_origin）一律用觀測值。雨天以**觀測**降雨 ≥ 1 mm 定義，各情境比較的是同一批小時。"
          "over_vs_a = 該情境 MAE − 情境 (a)；rain_gain_kept_% = 該情境保留了多少「天氣特徵在雨天帶來的 MAE 改善」"
          "（(d) − 情境）/（(d) − (a)）；rain_bias = 雨天平均（預測 − 實際）。\n",
          "## 預報本身的準確度\n", R.md_table(sk, "{:.3g}"), "", "## 模型誤差\n", R.md_table(df, "{:.2f}")]
    (RESULTS / f"reg_weather_forecast{suffix}.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
