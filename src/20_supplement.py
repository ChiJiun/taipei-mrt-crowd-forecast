"""Workstream B supplementary breakdowns of the main test-set predictions (no training; run after
20_regression.py, which writes results/pred_reg_h{h}_{target}.parquet).

  * MAE after excluding one-direction series: the open OD data splits some stations into entry-only /
    exit-only names (e.g. 大坪林 vs G大坪林) and Circular-line transfer gates only have entries, so 17 of the
    122 stations have an all-zero entries or exits series. Those trivially-zero rows lower every method's MAE.
  * MAE by station type (workstream C's K-means clusters, results/cluster_assignments.csv).
  * MAE on event-affected station-hours (events_station_hour.csv) and typhoon / holiday hours.

Usage: python src/20_supplement.py
Writes results/reg_supplement.csv, results/reg_supplement.md
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import RESULTS  # noqa: E402
from features import load_panel  # noqa: E402
from metrics import peak_mask  # noqa: E402

R = importlib.import_module("20_reglib")
METHODS = ["Seasonal Naive", "Historical Average", "LightGBM", "LightGBM-scaled", "LightGBM-B"]


def main():
    panel = load_panel()
    tot = panel.groupby("station", observed=True)[["entries", "exits"]].sum()
    cl_f = RESULTS / "cluster_assignments.csv"
    cl = pd.read_csv(cl_f).set_index("station")["cluster_name"] if cl_f.exists() else None
    rows = []
    for h in (1, 2):
        for tgt in ("entries", "exits"):
            f = RESULTS / f"pred_reg_h{h}_{tgt}.parquet"
            if not f.exists():
                continue
            p = pd.read_parquet(f)
            zero = set(tot.index[tot[tgt] == 0])
            st = p["station"].astype(str)
            subsets = {"全部站": np.ones(len(p), bool),
                       f"排除單向全 0 序列（{len(zero)} 站）": ~st.isin(zero).to_numpy()}
            if cl is not None:
                for c in sorted(cl.unique()):
                    subsets[f"站型：{c}"] = (st.map(cl) == c).to_numpy() & ~st.isin(zero).to_numpy()
            if "event_flag" in p:
                subsets["活動影響站時"] = (p["event_flag"] > 0).to_numpy()
            subsets["平日尖峰"] = peak_mask(p)
            subsets["假日"] = (p["is_holiday"] == 1).to_numpy()
            for name, m in subsets.items():
                if m.sum() == 0:
                    continue
                g = p[m]
                r = {"target": tgt, "horizon": h, "subset": name, "n": int(m.sum()), "mean_y": round(g.y.mean(), 1)}
                for mt in METHODS:
                    r[mt] = round(float(np.abs(g[mt] - g.y).mean()), 2)
                r["B_vs_HA_%"] = round(100 * (1 - r["LightGBM-B"] / r["Historical Average"]), 1)
                r["B_vs_scaled_%"] = round(100 * (1 - r["LightGBM-B"] / r["LightGBM-scaled"]), 1)
                rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "reg_supplement.csv", index=False)
    md = ["# 迴歸補充分析（測試集 2026-09）\n",
          "單向全 0 序列：OD 資料將部分車站拆成只有進站／只有出站的名稱，環狀線轉乘口只有進站，"
          "這些站的另一方向恆為 0，會讓所有方法的整體 MAE 偏低；排除後的數字較能代表一般車站。"
          "站型取自工作流 C 的 K-means 分群（已排除全 0 序列）。\n"]
    for (tgt, h), g in df.groupby(["target", "horizon"], sort=False):
        md += [f"## {'進站' if tgt == 'entries' else '出站'} t+{h}\n", R.md_table(g.drop(columns=["target", "horizon"])), ""]
    (RESULTS / "reg_supplement.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
