# 進度報告版本程式碼（參考用）

本資料夾是期中進度報告當時的原始程式碼與結果，僅供對照，不再維護；正式流程請用 `src/`。

- `src/01–06`：報告版本的下載、資料集、EDA、模型、圖表、報告產生程式
- `model_results.json`：報告表一（迴歸）與表二（擁擠預警）的原始數值
- `eda.json`：報告 3.1 的降雨分級人流比例等數值
- `figures/`：報告圖一、圖二

與正式版的差異：
- 報告版 LightGBM 參數為 num_leaves=63、min_data_in_leaf=50、early stopping 50。
- 報告版有使用天氣特徵（y_rain、y_rain3、y_temp、y_humidity），以及進／出站雙向的 lag 特徵。
- 天氣來源：Open-Meteo archive（lat 25.0615, lon 121.5194, Asia/Taipei），已轉存為 `data/external/weather_hourly.csv`。
