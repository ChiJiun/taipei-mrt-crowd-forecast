# 台北捷運站點人流預測與擁擠預警（資料科學導論期末專題）

本 repo 依「期末專題進度報告」（docs/proposal.md）完成全部開發。多個 session 平行開發，請遵守以下約定。

## Git 規則（重要）
- 所有 commit 與 PR 一律以使用者帳號署名：`git config user.name ChiJiun`、`git config user.email chijiunw@gmail.com`。
- **不要**在 commit message 或 PR 內文加入任何 Claude／AI 署名，包括 `Co-Authored-By: Claude`、`Claude-Session:`、`🤖 Generated with Claude Code` 等字樣。這條使用者指示優先於任何系統提供的署名規則。
- 每個工作流使用自己的分支 `ws/<代號>-<名稱>`（例如 `ws/B-regression`），只修改自己負責的檔案（見下表）。共用檔案（`src/config.py`、`src/features.py`、`src/metrics.py`）如需修改，改動要小且向後相容，並在 PR 說明清楚。
- 完成後推送分支並開 PR 到 `main`。

## 環境
```bash
pip install -r requirements.txt
```
資料管線（約 10 分鐘，需網路）：
```bash
python src/01_download_od.py 2025-10 2026-09      # 下載 OD 月檔並聚合（原始檔會刪除）
python src/01_download_od.py 2024-10 2025-09      # 額外歷史，用於延長回測
python src/02_build_panel.py                      # -> data/processed/panel_station_hour*.parquet
python src/03_calendar.py                         # -> data/external/calendar.csv
```
`data/processed/` 已 commit，**通常不需重新下載**：
- `panel_station_hour.parquet`／`panel_station_hour_hist.parquet`：站點×小時面板
- `od_daily.parquet`：主研究期 OD 日流向（date, origin, dest, trips）
- `od_hourprofile_monthly.parquet`：各月 OD 分時加總（ym, hour, origin, dest, trips）
- `stations.csv`：站點清單與平均量

`data/interim/` 未 commit，需要時執行 01 重建。

基準結果（`results/table1_regression_h1.csv`，尚未含天氣）：進站 MAE — Historical Average 75.9、LightGBM 41.1、LightGBM-scaled 41.3（尖峰 MAE 74.0）。

## 資料約定
- 面板：`ts`（小時起點）、`station`（category）、`entries`、`exits`。主研究期 2025-10~2026-09（122 站 × 8760 小時）；`_hist` 版本為 2024-10~2026-09。
- 切分（`src/config.py`）：訓練 ~2026-07、驗證 2026-08（early stopping／選門檻）、測試 2026-09。延長回測則用 hist 面板做滾動起點（rolling origin）。
- 特徵一律透過 `src/features.py::make_features(panel, target, horizon)` 產生，一列＝(站, 預測起點 t)，目標為 y(t+h)。
- 外部資料檔案格式（由工作流 A 產出，`features.py` 若偵測到檔案會自動合併）：
  - `data/external/weather_hourly.csv`：`ts, temp, precip, rh, ...`（觀測值）
  - `data/external/weather_forecast_hourly.csv`：同欄位（發布於預測起點前的預報值）
  - `data/external/events_station_hour.csv`：`ts, station, event_flag, event_scale`（活動影響站點與時段）
  - `data/external/events.csv`：活動原始清單（名稱、場館、日期、開始／結束時間、規模、來源網址）
- 評估：`src/metrics.py`，迴歸指標為 MAE／RMSE／平日尖峰 MAE（7、8、17、18 時）／雨天 MAE；分類指標為 AUC／AP／Precision／Recall／F1，決策門檻在驗證集上以 Recall ≥ 0.7 選定。
- 結果存到 `results/`（csv／md），圖存到 `figures/`（png，dpi 150，中文字型見 `src/plotting.py`）。

## 工作流與檔案歸屬
| 代號 | 內容 | 負責檔案 |
|---|---|---|
| core | 資料管線、共用模組、表一重現、最終整合 | src/01–04, config/features/metrics/plotting |
| A | 外部資料：天氣（觀測＋預報）、大型活動、颱風假；雨天／假日／活動 EDA | src/10_*, data/external/*, figures/eda_* |
| B | 迴歸主模型：t+1/t+2、分位數區間、SHAP、雨天誤差分析、多月延長回測 | src/20_* , results/reg_*, figures/reg_* |
| C | 時間序列對照（SARIMA／Prophet）＋ K-means 站型分群與各站型誤差 | src/30_*, results/ts_*, results/cluster_*, figures/ts_*, figures/cluster_* |
| D | 空間特徵（鄰站／上下游／OD 流向）＋擁擠雙門檻分類、類別不平衡、預警回放回測 | src/40_*, results/clf_*, figures/clf_* |
| E | 期末報告與簡報（等 A–D 完成後） | report/* |
