"""結果圖：台北車站一週預測 vs 實際"""
import os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, pandas as pd
plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "sans-serif"]; plt.rcParams["axes.unicode_minus"] = False
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
t = pd.read_parquet(f"{ROOT}/results/test_pred.parquet")
s = t[(t.station == "台北車站") & (t.y_ts >= "2026-09-14") & (t.y_ts < "2026-09-21")]
fig, ax = plt.subplots(figsize=(7, 2.6))
ax.plot(s.y_ts, s.y_entry, label="實際進站量", lw=1.2); ax.plot(s.y_ts, s.p_entry, label="模型預測", lw=1.2, ls="--")
ax.set_ylabel("人次/小時"); ax.legend(fontsize=8); ax.grid(alpha=.3); fig.autofmt_xdate(); fig.tight_layout()
fig.savefig(f"{ROOT}/figures/fig2_taipei_main_week.png", dpi=200)
