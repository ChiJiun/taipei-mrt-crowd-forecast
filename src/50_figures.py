"""工作流 E：期末報告／簡報用的整合圖（figures/final_*.png）。

  final_method_mae.png  各方法進站 MAE（t+1、t+2）比較
  final_goals.png       提案「預期成果」達成總覽（MAE 改善 ≥ 25%；Recall ≥ 0.7 時 Precision ≥ 0.9）
  final_pipeline.png    資料與方法流程圖

數字全部讀自 results/（reg_main.csv、ts_compare.csv、clf_main.csv）。
用法：python src/50_figures.py        （草稿階段可加環境變數 REPORT_REF=origin/ws/B-regression）
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from plotting import plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

rd = importlib.import_module("50_report_data")
FIG = rd.FIGURES

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#b8b6ae"


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)


def method_table():
    ts = rd.read_csv("ts_compare.csv")
    ts = ts[ts.target == "entries"]
    rows = {}
    for h in (1, 2):
        d = ts[ts.horizon == h].set_index("method")["MAE"].to_dict()
        rows[h] = {k: d[k] for k in ["Seasonal Naive", "Historical Average", "Prophet", "Prophet+AR", "SARIMAX"]}
        if rd.has("reg_main.csv"):
            rm = rd.read_csv("reg_main.csv")
            rm = rm[(rm.target == "entries") & (rm.horizon == h)].set_index("method")["MAE"].to_dict()
            for k in ["LightGBM", "LightGBM-scaled", "LightGBM-B"]:
                rows[h][k] = rm[k]
        else:
            rows[h]["LightGBM"] = d["LightGBM"]
            rows[h]["LightGBM-scaled"] = d["LightGBM (scaled)"]
    return rows


def fig_method_mae():
    rows = method_table()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharex=True)
    for ax, h in zip(axes, (1, 2)):
        items = sorted(rows[h].items(), key=lambda kv: -kv[1])
        names = [k for k, _ in items]
        vals = [v for _, v in items]
        best = names[-1]
        colors = [BLUE if n == best else GRAY for n in names]
        ax.barh(names, vals, color=colors, height=0.62, edgecolor="white", linewidth=2)
        for i, v in enumerate(vals):
            ax.text(v + 1.5, i, f"{v:.1f}", va="center", fontsize=8.5,
                    bbox=dict(fc="white", ec="none", pad=0.6),
                    color=INK if names[i] == best else INK2, fontweight="bold" if names[i] == best else None)
        ha = rows[h]["Historical Average"]
        ax.axvline(ha * 0.75, color=INK2, lw=1, ls="--")
        ax.text(ha * 0.75, len(names) - 0.4, " HA×0.75\n（改善 25%）", fontsize=7.5, color=INK2, va="top")
        ax.set_title(f"進站 t+{h}", fontsize=11, color=INK, loc="left")
        ax.set_xlabel("測試集 MAE（人次／小時）", fontsize=9, color=INK2)
        ax.xaxis.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        _style(ax)
    fig.suptitle("各方法進站預測誤差（測試集 2026-09，越短越好）", fontsize=12, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(FIG / "final_method_mae.png", dpi=150)
    plt.close(fig)


def fig_goals():
    model = "LightGBM-B" if rd.has("reg_main.csv") else "LightGBM (scaled)"
    if rd.has("reg_main.csv"):
        rm = rd.read_csv("reg_main.csv")
        imp = {(t, h): rm[(rm.target == t) & (rm.horizon == h) & (rm.method == model)]["MAE_vs_HA_%"].iloc[0]
               for t in ("entries", "exits") for h in (1, 2)}
    else:
        ts = rd.read_csv("ts_compare.csv")
        imp = {}
        for t in ("entries", "exits"):
            for h in (1, 2):
                d = ts[(ts.target == t) & (ts.horizon == h)].set_index("method")["MAE"]
                imp[(t, h)] = 100 * (1 - d[model] / d["Historical Average"])
    clf = rd.read_csv("clf_main.csv")
    clf = clf[(clf.method == "LGB-clf[none]") & (clf.group == "全部")].set_index("label")

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    ax = axes[0]
    labels = ["進站 t+1", "出站 t+1", "進站 t+2", "出站 t+2"]
    vals = [imp[("entries", 1)], imp[("exits", 1)], imp[("entries", 2)], imp[("exits", 2)]]
    ax.bar(labels, vals, color=BLUE, width=0.6, edgecolor="white", linewidth=2)
    for i, v in enumerate(vals):
        ax.text(i, v + 1, f"{v:.1f}%", ha="center", fontsize=9, color=INK)
    ax.axhline(25, color=ORANGE, lw=1.5, ls="--")
    ax.text(-0.45, 26, "目標 25%", color=INK2, fontsize=8.5, ha="left", va="bottom")
    ax.set_ylim(0, max(vals) * 1.25)
    ax.set_ylabel("MAE 相對 HA 改善（%）", fontsize=9, color=INK2)
    ax.set_title(f"(a) 迴歸：{model} vs Historical Average", fontsize=10, color=INK, loc="left")
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    _style(ax)

    ax = axes[1]
    defs = [("relative", "相對 P95"), ("absolute", "絕對門檻"), ("dual", "雙門檻")]
    x = range(len(defs))
    p = [clf.loc[k, "Precision"] for k, _ in defs]
    r = [clf.loc[k, "Recall"] for k, _ in defs]
    w = 0.36
    ax.bar([i - w / 2 for i in x], p, w, color=BLUE, label="Precision", edgecolor="white", linewidth=2)
    ax.bar([i + w / 2 for i in x], r, w, color=ORANGE, label="Recall", edgecolor="white", linewidth=2)
    for i in x:
        ax.text(i - w / 2, p[i] + 0.01, f"{p[i]:.2f}", ha="center", fontsize=8.5, color=INK)
        ax.text(i + w / 2, r[i] + 0.01, f"{r[i]:.2f}", ha="center", fontsize=8.5, color=INK)
    ax.axhline(0.9, color=BLUE, lw=1, ls="--")
    ax.axhline(0.7, color=ORANGE, lw=1, ls="--")
    ax.text(2.62, 0.905, "P 目標 0.9", fontsize=8, color=INK2, ha="right", va="bottom")
    ax.text(2.6, 0.705, "R 目標 0.7", fontsize=8, color=INK2, ha="right", va="bottom")
    ax.set_xticks(list(x), [n for _, n in defs])
    ax.set_ylim(0.5, 1.16)
    ax.set_title("(b) 擁擠預警：LightGBM 分類（t+1）", fontsize=10, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", ncol=2)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    _style(ax)
    fig.suptitle("提案預期成果達成情形（測試集 2026-09）", fontsize=12, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(FIG / "final_goals.png", dpi=150)
    plt.close(fig)


def fig_pipeline():
    fig, ax = plt.subplots(figsize=(10, 4.4))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 46)
    ax.axis("off")

    def box(x, y, w, h, title, body, fc):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                    fc=fc, ec="#8f8d85", lw=1))
        ax.text(x + w / 2, y + h - 1.8, title, ha="center", va="top", fontsize=10, fontweight="bold", color=INK)
        ax.text(x + w / 2, y + h - 6.2, body, ha="center", va="top", fontsize=7.6, color=INK2, linespacing=1.4)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color="#6d6b64", lw=1.2))

    light = "#eef4fc"
    box(1, 25, 19, 19, "原始資料", "北捷 OD 月檔（24 個月）\nOpen-Meteo 逐時天氣\n人事行政總處日曆\n大型活動 574 場次\n颱風停班停課日", "#f4f3ef")
    box(1, 2, 19, 18, "資料處理", "逐月聚合後刪原始檔\n站 × 小時面板\n122 站 × 8,760 小時", "#f4f3ef")
    box(24, 2, 20, 42, "特徵（features.py）", "時段、星期、假日\n補班、連假長度\n延遲 1/2/3/24/48/168/336h\n滾動平均\n歷史平均剖面\n天氣（觀測／預報）\n活動旗標與規模\n鄰站、同線上下游\nOD 流向傳遞", light)
    box(48, 25, 23, 19, "迴歸（t+1、t+2）", "基準：LV／SN／HA\n時序：SARIMAX、Prophet\nLightGBM／-scaled／-B\n分位數 P10–P90 區間", light)
    box(48, 2, 23, 18, "分類（擁擠預警）", "相對 P95／絕對／雙門檻\n迴歸÷門檻、二元分類\n類別不平衡處理", light)
    box(75, 25, 24, 19, "評估", "訓練 ~2026-07\n驗證 2026-08、測試 2026-09\nMAE／RMSE／尖峰／雨天\nSHAP、站型（K-means）", "#fdf1ea")
    box(75, 2, 24, 18, "回測", "多月滾動起點（2024-10 起）\n預警回放：提前量、漏報\n誤報成本、非例行事件", "#fdf1ea")
    arrow(10.5, 24.5, 10.5, 20.6)
    arrow(20.6, 11, 23.4, 11)
    arrow(44.6, 34, 47.4, 34)
    arrow(44.6, 11, 47.4, 11)
    arrow(71.6, 34, 74.4, 34)
    arrow(71.6, 11, 74.4, 11)
    fig.savefig(FIG / "final_pipeline.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    fig_method_mae()
    fig_goals()
    fig_pipeline()
    print("wrote figures/final_method_mae.png, final_goals.png, final_pipeline.png")
