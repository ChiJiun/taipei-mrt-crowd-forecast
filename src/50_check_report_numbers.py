"""工作流 E：抽查 report/final_report.md 的關鍵數字是否與 results/ 的檔案一致。

每一項檢查 = （結果檔, 篩選條件, 欄位, 報告中應出現的字串樣板）。
程式從結果檔取出數值、依樣板格式化，再確認該字串確實出現在報告中。
結果檔不存在時標為 SKIP（例如工作流 B 尚未合併）；草稿階段可設定
REPORT_REF=origin/ws/B-regression 從該分支讀取。

用法：python src/50_check_report_numbers.py [--strict]   （--strict：有 SKIP 也回傳非 0）
"""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
rd = importlib.import_module("50_report_data")
REPORT = rd.ROOT / "report" / "final_report.md"


def val(df, query, col):
    sub = df.query(query) if query else df
    if len(sub) != 1:
        raise ValueError(f"篩選 {query!r} 得到 {len(sub)} 列")
    return sub[col].iloc[0]


# (file, query, column, template, scale)
T1 = "table1_regression_h1.csv"
RM = "reg_main.csv"
TS = "ts_compare.csv"
CE = "cluster_error.csv"
SA = "spatial_ablation.csv"
CM = "clf_main.csv"
CT = "clf_table2.csv"
BT = "clf_backtest.csv"
RS = "reg_supplement.csv"
SG = "reg_shap_groups.csv"

CHECKS = [
    # 表二：表一重現
    (T1, "target=='entries' and method=='LightGBM (scaled)'", "MAE", "LightGBM-scaled 進站 MAE {v:.1f}"),
    (T1, "target=='entries' and method=='LightGBM (scaled)'", "peak_MAE", "平日尖峰 MAE {v:.1f}"),
    (T1, "target=='entries' and method=='LightGBM (scaled)'", "rain_MAE", "雨天 MAE {v:.1f}，"),
    (T1, "target=='entries' and method=='LightGBM (scaled)'", "dry_MAE", "無雨 {v:.1f}）"),
    (T1, "target=='entries' and method=='Historical Average'", "MAE", "| Historical Average | {v:.1f} |"),
    (T1, "target=='exits' and method=='LightGBM (scaled)'", "MAE", "| LightGBM (scaled) | 41.9 | 117.6 | 77.5 | {v:.1f} |"),
    (T1, "target=='exits' and method=='LightGBM'", "peak_MAE", "| 117.7 | {v:.1f} |"),
    # 表三：迴歸主結果
    (RM, "target=='entries' and horizon==1 and method=='LightGBM-B'", "MAE", "| 進站 t+1 | LightGBM-B | {v:.1f} |"),
    (RM, "target=='entries' and horizon==1 and method=='LightGBM-B'", "MAE_vs_HA_%", "（相對 HA 改善 {v:.1f}%）"),
    (RM, "target=='exits' and horizon==1 and method=='LightGBM-B'", "MAE", "| 出站 t+1 | LightGBM-B | {v:.1f} |"),
    (RM, "target=='exits' and horizon==1 and method=='LightGBM-B'", "MAE_vs_HA_%", "出站 37.0（{v:.1f}%）"),
    (RM, "target=='entries' and horizon==2 and method=='LightGBM-B'", "MAE", "| 進站 t+2 | LightGBM-B | {v:.1f} |"),
    (RM, "target=='entries' and horizon==2 and method=='LightGBM-B'", "MAE_vs_HA_%", "t+2 進站 41.6（{v:.1f}%）"),
    (RM, "target=='exits' and horizon==2 and method=='LightGBM-B'", "MAE", "| 出站 t+2 | LightGBM-B | {v:.1f} |"),
    (RM, "target=='exits' and horizon==2 and method=='LightGBM-B'", "MAE_vs_HA_%", "出站 41.1（{v:.1f}%）"),
    (RM, "target=='entries' and horizon==1 and method=='LightGBM-scaled'", "MAE", "| 進站 t+1 | LightGBM-scaled | {v:.1f} |"),
    (RM, "target=='entries' and horizon==2 and method=='LightGBM-scaled'", "MAE", "| 進站 t+2 | LightGBM-scaled | {v:.1f} |"),
    (RM, "target=='entries' and horizon==2 and method=='LightGBM-scaled'", "peak_MAE", "尖峰 MAE 也從 {v:.1f} 降到"),
    (RM, "target=='entries' and horizon==2 and method=='LightGBM-B'", "peak_MAE", "降到 {v:.1f}，"),
    (RM, "target=='entries' and horizon==1 and method=='LightGBM-B'", "rain_MAE", "雨天 MAE {v:.1f} vs 無雨"),
    (RM, "target=='entries' and horizon==1 and method=='LightGBM-B'", "dry_MAE", "vs 無雨 {v:.1f}"),
    (RM, "target=='entries' and horizon==1 and method=='LightGBM-scaled'", "MAE_vs_HA_%", "LightGBM-scaled {v:.1f}%、"),
    # 表四：時間序列對照
    (TS, "target=='entries' and horizon==1 and method=='SARIMAX'", "MAE", "| SARIMAX | {v:.1f} |"),
    (TS, "target=='entries' and horizon==1 and method=='SARIMAX'", "rain_MAE", "（{v:.1f} vs 72.1）"),
    (TS, "target=='entries' and horizon==2 and method=='SARIMAX'", "MAE", "| 102.2 | 71.0 | {v:.1f} |"),
    (TS, "target=='entries' and horizon==1 and method=='Prophet'", "MAE", "| Prophet | {v:.1f} |"),
    (TS, "target=='entries' and horizon==1 and method=='Prophet+AR'", "MAE", "| Prophet+AR | {v:.1f} |"),
    (TS, "target=='entries' and horizon==2 and method=='LightGBM (scaled)'", "MAE", "| 72.1 | {v:.1f} |"),
    # 表五：站型
    (CE, "target=='entries' and horizon==1 and cluster_name=='轉乘樞紐型' and method=='LightGBM (scaled)'", "rel_MAE",
     "| 台北車站、西門、中山 | {v:.3f} |"),
    (CE, "target=='entries' and horizon==1 and cluster_name=='商圈觀光型' and method=='LightGBM (scaled)'", "rel_MAE",
     "| 市政府、台北101/世貿、士林 | {v:.3f} |"),
    (CE, "target=='entries' and horizon==1 and cluster_name=='住宅型' and method=='LightGBM (scaled)'", "rel_MAE",
     "| 新埔、頂溪、淡水 | {v:.3f} |"),
    # 表六：空間特徵
    (SA, "target=='entries' and horizon==2 and features=='+all'", "MAE", "| 進站 t+2 | 46.02 | – | – | {v:.2f}（"),
    (SA, "target=='exits' and horizon==2 and features=='+all'", "MAE_change_vs_base_%", "42.53（−{v:.1f}%）", ),
    (SA, "target=='entries' and horizon==2 and features=='+all'", "rain_MAE", "104.6 → {v:.1f}"),
    # 表七：擁擠預警
    (CM, "label=='relative' and method=='LGB-clf[none]' and group=='全部'", "Precision", "| 相對 P95 | LGB 分類 | 0.998 | 0.973 | {v:.3f} |"),
    (CM, "label=='relative' and method=='LGB-clf[none]' and group=='全部'", "Recall", "| 0.965 | {v:.3f} | 0.859 |"),
    (CM, "label=='absolute' and method=='LGB-clf[none]' and group=='全部'", "Recall", "| 1.000 | {v:.3f} | 0.834 |"),
    (CM, "label=='dual' and method=='LGB-clf[none]' and group=='全部'", "Recall", "| 0.965 | {v:.3f} | 0.890 |"),
    (CM, "label=='dual' and method=='HistAvg' and group=='全部'", "Recall", "| 0.959 | {v:.3f} |"),
    (CT, "method=='LightGBM-binary'", "Precision", "Precision {v:.3f}／Recall 0.745"),
    # 表八：回放回測
    (BT, "period=='2026-09' and policy=='min cost 10:1 (valid)'", "prewarn_rate", "| 4,078 | {v:.1%} |"),
    (BT, "period=='2026-01' and policy=='min cost 10:1 (valid)'", "prewarn_nonroutine", "| 234 | {v:.1%} |"),
    (BT, "period=='2026-02' and policy=='min cost 10:1 (valid)'", "fa_per_station_day", "| 25 | {v:.2f} |"),
    (BT, "period=='2026-07' and policy=='min cost 10:1 (valid)'", "prewarn_nonroutine", "| 143 | {v:.1%} |"),
    (BT, "period=='2026-07' and policy=='min cost 10:1 (valid)'", "prewarn_rate", "| 3,551 | {v:.1%} |"),
    ("clf_backtest_typhoon.csv", "subset=='颱風停班日' and policy=='min cost 10:1 (valid)'", "fa_per_station_day", "只有 4 個誤報小時（{v:.3f}／站／日"),
    ("clf_backtest_typhoon.csv", "subset=='其餘日' and policy=='min cost 10:1 (valid)'", "fa_per_station_day", "同月其餘日為 {v:.2f}）"),
    # 補充：排除全 0 序列、活動站時
    (RS, "target=='entries' and horizon==1 and subset.str.startswith('排除')", "LightGBM-B", "t+1 進站 MAE 為 {v:.1f}"),
    (RS, "target=='entries' and horizon==1 and subset=='活動影響站時'", "B_vs_HA_%", "改善為 {v:.0f}%（進站）"),
    # SHAP
    (SG, "group=='歷史平均剖面'", "share_%", "歷史平均剖面占平均 |SHAP| 的 {v:.1f}%"),
    (SG, "group=='天氣'", "share_%", "天氣 {v:.1f}%"),
    # 多月回測（迴歸）
    ("reg_backtest_special.csv", "target=='entries' and subset=='颱風假'", "MAE_LightGBM-B", "| 進站：颱風假（2 天） | 629.9 | 560.6 | 196.9 | {v:.1f} |"),
    ("reg_backtest_special.csv", "target=='entries' and subset=='颱風假'", "bias_LightGBM-B", "平均偏差 +{v:.0f}"),
    ("reg_backtest_special.csv", "target=='entries' and subset=='春節連假'", "MAE_LightGBM-B", "| 167.6 | 53.8 | {v:.1f} |"),
    ("reg_backtest_special.csv", "target=='entries' and subset.str.startswith('跨年')", "MAE_LightGBM-B", "LightGBM-B（{v:.1f}）甚至略遜"),
    # 分位數區間
    ("reg_quantile.csv", "target=='entries' and horizon==1 and interval=='CQR-分組' and subset=='all'", "coverage", "| 進站 t+1 | {v:.1%} |"),
    ("reg_quantile.csv", "target=='entries' and horizon==1 and interval=='CQR-分組' and subset=='peak'", "coverage", "| 79.7% | {v:.1%} |"),
    ("reg_quantile.csv", "target=='exits' and horizon==2 and interval=='CQR-分組' and subset=='rain'", "coverage", "| 74.8% | {v:.1%} |"),
    ("reg_quantile.csv", "target=='exits' and horizon==1 and interval=='raw' and subset=='all'", "coverage", "未校準時出站 t+1 只有 {v:.1%}"),
    ("reg_quantile.csv", "target=='exits' and horizon==1 and interval=='CQR-分組' and subset=='all'", "coverage", "CQR 後提升到 {v:.1%}"),
    # 消融（天氣上限）
    ("reg_ablation.csv", "target=='entries' and horizon==1 and method=='LightGBM-B (no weather)'", "MAE", "整體 MAE 為 {v:.2f}"),
    ("reg_ablation.csv", "target=='entries' and horizon==1 and method=='LightGBM-B (no weather)'", "rain_MAE", "雨天 MAE 從 60.7 升到 {v:.1f}，因此"),
]


def eda_checks(report: str):
    out = []
    eda = json.loads(rd.read_text("eda_summary_numbers.json"))
    rr = rd.read_csv("eda_rain_ratio.csv")

    def r(period, dt, rain):
        x = rr[(rr.period == period) & (rr.day_type == dt) & (rr.rain == rain)]
        assert len(x) == 1, (period, dt, rain)
        return x.iloc[0]
    P2, PM, PT = "2024-10~2026-09", "2025-10~2026-09", "2025-10~2026-09 含颱風停班日（=進度報告定義）"
    x = r(P2, "假日", "1–5mm")
    out.append(("eda_rain_ratio.csv", "2y 假日中雨", f"假日全網進站量降到無雨的 {x.ratio_mean * 100:.0f}%（CI {x.ci_low:.2f}–{x.ci_high:.2f}）"))
    x = r(P2, "平日", "1–5mm")
    out.append(("eda_rain_ratio.csv", "2y 平日中雨", f"平日 {x.ratio_mean * 100:.0f}%（CI {x.ci_low:.2f}–{x.ci_high:.2f}）；"))
    x = r(P2, "平日", ">5mm")
    out.append(("eda_rain_ratio.csv", "2y 平日大雨", f"平日為 {x.ratio_mean * 100:.0f}%（CI {x.ci_low:.2f}–{x.ci_high:.2f}）"))
    x = r(P2, "假日", ">5mm")
    out.append(("eda_rain_ratio.csv", "2y 假日大雨", f"假日為 {x.ratio_mean * 100:.0f}%（CI {x.ci_low:.2f}–{x.ci_high:.2f}，只有 {x.n_hours} 個小時、{x.n_days} 天）"))
    out.append(("eda_rain_ratio.csv", "2y 假日大雨（限制）", f"兩年資料也只有 {x.n_hours} 個小時（CI {x.ci_low:.2f}–{x.ci_high:.2f}）"))
    x = r(PT, "假日", ">5mm")
    out.append(("eda_rain_ratio.csv", "進度報告定義", f"{x.ratio_mean:.2f}，CI {x.ci_low:.2f}–{x.ci_high:.2f}，{x.n_hours} 小時、{x.n_days} 天"))
    out.append(("eda_rain_ratio.csv", "進度報告定義（限制）", f"信賴區間寬達 {x.ci_low:.2f}–{x.ci_high:.2f}"))
    x = r(PM, "假日", ">5mm")
    out.append(("eda_rain_ratio.csv", "主研究期排除颱風", f"假日 > 5 mm 為 {x.ratio_mean * 100:.0f}%（CI {x.ci_low:.2f}–{x.ci_high:.2f}，只有 {x.n_hours} 個小時）"))
    a, b = r(PM, "假日", "1–5mm"), r(PM, "平日", "1–5mm")
    out.append(("eda_rain_ratio.csv", "主研究期中雨", f"中雨時假日 {a.ratio_mean * 100:.0f}%、平日 {b.ratio_mean * 100:.0f}%"))
    fv = eda["forecast_vs_obs"]
    out.append(("eda_summary_numbers.json", "預報命中率", f"命中率 {fv['rain_hit_rate']:.2f}、誤報率 {fv['rain_false_alarm_ratio']:.2f}"))
    out.append(("eda_summary_numbers.json", "預報 MAE", f"氣溫 MAE {fv['temp_mae']:.2f} °C、降雨 MAE {fv['precip_mae']:.2f} mm"))
    out.append(("eda_summary_numbers.json", "假日／平日", f"假日／平日 = {eda['holiday_to_workday_ratio']:.2f}"))
    bt = rd.read_csv("reg_backtest_monthly.csv")
    x = bt[bt.target == "entries"].pivot(index="month", columns="method", values="MAE")
    imp = 100 * (1 - x["LightGBM-B"] / x["Historical Average"])
    out.append(("reg_backtest_monthly.csv", "各月改善最小", f"介於 {imp.min():.1f}%（{imp.idxmin()}）與 {imp.max():.1f}%（{imp.idxmax()}）"))
    out.append(("reg_backtest_monthly.csv", "2026-09 改善", f"主測試月 2026-09 為 {imp['2026-09']:.1f}%"))
    n = bt[bt.target == "entries"].groupby("method").apply(lambda g: (g.MAE * g.n).sum() / g.n.sum())
    out.append(("reg_backtest_monthly.csv", "全年 MAE", f"| 進站全年 MAE | {n['Seasonal Naive']:.1f} | {n['Historical Average']:.1f} | "
                f"{n['LightGBM-scaled']:.1f} | {n['LightGBM-B']:.1f} |"))
    import re as _re
    t = rd.read_text("reg_rain_tables.md")
    sec = t[t.index("控制組成後"):]
    v = [float(m) for m in _re.findall(r"\| ([\d.]+) \|", sec)[:3]]
    out.append(("reg_rain_tables.md", "雨天 MAE", f"雨天 MAE {v[0]:.1f}，無雨 {v[1]:.1f}"))
    out.append(("reg_rain_tables.md", "組成加權", f"無雨 MAE 升到 {v[2]:.1f}"))
    return [(f, d, s, s in report) for f, d, s in out]


def main():
    strict = "--strict" in sys.argv
    report = REPORT.read_text(encoding="utf-8")
    ok = bad = skip = 0
    cache = {}
    for f, q, col, tmpl in CHECKS:
        if not rd.has(f):
            print(f"SKIP  {f}（檔案不存在）  {tmpl}")
            skip += 1
            continue
        if f not in cache:
            cache[f] = rd.read_csv(f)
        try:
            v = val(cache[f], q, col)
            s = tmpl.format(v=abs(v) if "−{v" in tmpl else v)
        except Exception as e:  # noqa: BLE001
            print(f"ERROR {f} [{q}] {col}: {e}")
            bad += 1
            continue
        src = rd.source_of(f)
        tag = "" if src == "local" else f"  (from {src})"
        if s in report:
            ok += 1
            print(f"OK    {f}: {s}{tag}")
        else:
            bad += 1
            print(f"FAIL  {f}: 報告中找不到「{s}」{tag}")
    for f, d, s, found in eda_checks(report):
        if found:
            ok += 1
            print(f"OK    {f}: {s}")
        else:
            bad += 1
            print(f"FAIL  {f} {d}: 報告中找不到「{s}」")
    print(f"\n通過 {ok}、不一致 {bad}、略過 {skip}")
    sys.exit(1 if bad or (strict and skip) else 0)


if __name__ == "__main__":
    main()
