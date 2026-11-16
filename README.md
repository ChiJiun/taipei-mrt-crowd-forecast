# 台北捷運站點人流預測與擁擠預警

資料科學導論期末專題。預測台北捷運各站未來 1–2 小時的進／出站量（迴歸），以及下一小時是否擁擠（分類），並分析雨天、假日與大型活動的影響。

- **期末報告**：[report/final_report.docx](report/final_report.docx)（原稿 [final_report.md](report/final_report.md)）
- **期末簡報**：[report/slides.pptx](report/slides.pptx)（PDF：[slides.pdf](report/slides.pdf)）
- 提案／進度報告：[docs/proposal.md](docs/proposal.md)
- 開發約定與工作流分工：[CLAUDE.md](CLAUDE.md)

## 執行順序
| 步驟 | 程式 | 產出 |
|---|---|---|
| 1 | `src/01_download_od.py` | 每月 OD → 站點×小時、OD 日流向（data/interim） |
| 2 | `src/02_build_panel.py` | `data/processed/panel_station_hour.parquet`（122 站 × 8760 小時） |
| 3 | `src/03_calendar.py` | `data/external/calendar.csv` |
| 4 | `src/04_baselines_t1.py` | 表一：基準方法 vs LightGBM（`results/table1_regression_h1.csv`） |
| 10–11 | 外部資料與 EDA | 天氣（觀測／預報）、活動、颱風假；`results/eda_summary.md` |
| 20 | 迴歸主模型 | t+1/t+2、分位數區間、SHAP、雨天分析、多月回測；`results/reg_summary.md` |
| 30–32 | 時間序列與站型 | SARIMAX／Prophet、K-means 站型；`results/ts_cluster_summary.md` |
| 40–43 | 空間特徵與擁擠預警 | 表二、雙門檻、預警回放；`results/clf_summary.md` |
| 50 | 期末報告與簡報 | `report/`（`src/50_check_report_numbers.py` 核對報告數字） |

## 資料來源
- 臺北大眾捷運公司「每日分時各站 OD 流量統計資料」（政府資料開放平臺 dataset 128506）
- Open-Meteo 歷史再分析／歷史預報；中央氣象署開放資料
- TaiwanCalendar（依人事行政總處辦公日曆整理）
