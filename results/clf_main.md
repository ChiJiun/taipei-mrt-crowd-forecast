# 擁擠預警主結果（t+1，測試集 2026-09）

- 特徵：features.py 基礎特徵 ＋ 空間特徵（41_spatial_features）
- 相對門檻：各站訓練期 (進+出) P95；絕對門檻 A = 全網站時（06–23 時）訓練期 P95 = 5914 人次/小時；雙門檻 = 兩者同時成立
- 測試期出現正類的站數：相對 122、絕對 45、雙門檻 45（共 122 站）
- 決策門檻：驗證集（2026-08）Recall ≥ 0.7 的最高門檻；target_met = 測試集 Recall ≥ 0.7 且 Precision ≥ 0.9

## 整體

| label | method | AUC | AP | Precision | Recall | F1 | thr | pos_rate | target_met |
|---|---|---|---|---|---|---|---|---|---|
| relative | HistAvg | 0.995 | 0.937 | 0.952 | 0.56 | 0.705 | 1.0349 | 0.0571 | False |
| relative | LGB-reg/thr | 0.998 | 0.972 | 0.951 | 0.825 | 0.883 | 1.0277 | 0.0571 | True |
| relative | LGB-clf[none] | 0.998 | 0.973 | 0.965 | 0.773 | 0.859 | 0.8698 | 0.0571 | True |
| relative | LGB-clf[class_weight] | 0.998 | 0.969 | 0.963 | 0.766 | 0.853 | 0.9865 | 0.0571 | True |
| relative | LGB-clf[focal] | 0.998 | 0.973 | 0.965 | 0.769 | 0.856 | 0.753 | 0.0571 | True |
| relative | LGB-clf[undersample] | 0.998 | 0.968 | 0.959 | 0.764 | 0.85 | 0.9601 | 0.0571 | True |
| absolute | HistAvg | 0.998 | 0.963 | 0.987 | 0.671 | 0.799 | 1.1296 | 0.0382 | False |
| absolute | LGB-reg/thr | 1.0 | 0.991 | 0.999 | 0.699 | 0.823 | 1.1432 | 0.0382 | False |
| absolute | LGB-clf[none] | 1.0 | 0.994 | 1.0 | 0.715 | 0.834 | 0.9943 | 0.0382 | True |
| absolute | LGB-clf[class_weight] | 1.0 | 0.994 | 1.0 | 0.722 | 0.838 | 0.9992 | 0.0382 | True |
| absolute | LGB-clf[focal] | 1.0 | 0.994 | 1.0 | 0.718 | 0.836 | 0.906 | 0.0382 | True |
| absolute | LGB-clf[undersample] | 1.0 | 0.993 | 0.999 | 0.71 | 0.83 | 0.9986 | 0.0382 | True |
| dual | HistAvg | 0.997 | 0.915 | 0.959 | 0.612 | 0.747 | 1.0221 | 0.0171 | False |
| dual | LGB-reg/thr | 0.999 | 0.97 | 0.947 | 0.871 | 0.907 | 1.0187 | 0.0171 | True |
| dual | LGB-clf[none] | 0.999 | 0.974 | 0.965 | 0.827 | 0.89 | 0.8103 | 0.0171 | True |
| dual | LGB-clf[class_weight] | 1.0 | 0.974 | 0.964 | 0.802 | 0.876 | 0.9938 | 0.0171 | True |
| dual | LGB-clf[focal] | 1.0 | 0.976 | 0.961 | 0.827 | 0.889 | 0.7164 | 0.0171 | True |
| dual | LGB-clf[undersample] | 0.999 | 0.968 | 0.954 | 0.8 | 0.87 | 0.9789 | 0.0171 | True |

## 正類分布（測試集）

| label | group | positives | pos_rate |
|---|---|---|---|
| relative | 中站(21-60) | 1600 | 0.0556 |
| relative | 大站(前20) | 754 | 0.0524 |
| relative | 小站(61+) | 2662 | 0.0596 |
| relative | 全部 | 5016 | 0.0571 |
| absolute | 中站(21-60) | 807 | 0.028 |
| absolute | 大站(前20) | 2531 | 0.1758 |
| absolute | 小站(61+) | 14 | 0.0003 |
| absolute | 全部 | 3352 | 0.0382 |
| dual | 中站(21-60) | 775 | 0.0269 |
| dual | 大站(前20) | 716 | 0.0497 |
| dual | 小站(61+) | 14 | 0.0003 |
| dual | 全部 | 1505 | 0.0171 |

## 依站點規模（測試集，Precision／Recall）

| label | method | Precision_中站(21-60) | Precision_大站(前20) | Precision_小站(61+) | Recall_中站(21-60) | Recall_大站(前20) | Recall_小站(61+) |
|---|---|---|---|---|---|---|---|
| absolute | HistAvg | 1.0 | 0.985 | 0.0 | 0.421 | 0.755 | 0.0 |
| absolute | LGB-clf[class_weight] | 1.0 | 0.999 | 0.0 | 0.601 | 0.764 | 0.0 |
| absolute | LGB-clf[focal] | 1.0 | 1.0 | 0.0 | 0.606 | 0.758 | 0.0 |
| absolute | LGB-clf[none] | 1.0 | 1.0 | 0.0 | 0.592 | 0.759 | 0.0 |
| absolute | LGB-clf[undersample] | 1.0 | 0.999 | 0.0 | 0.559 | 0.763 | 0.0 |
| absolute | LGB-reg/thr | 1.0 | 0.999 | 0.0 | 0.522 | 0.759 | 0.0 |
| dual | HistAvg | 0.972 | 0.946 | 0.0 | 0.627 | 0.608 | 0.0 |
| dual | LGB-clf[class_weight] | 0.979 | 0.948 | 1.0 | 0.823 | 0.782 | 0.643 |
| dual | LGB-clf[focal] | 0.974 | 0.947 | 0.917 | 0.834 | 0.82 | 0.786 |
| dual | LGB-clf[none] | 0.98 | 0.95 | 0.917 | 0.828 | 0.825 | 0.786 |
| dual | LGB-clf[undersample] | 0.977 | 0.93 | 0.9 | 0.817 | 0.785 | 0.643 |
| dual | LGB-reg/thr | 0.961 | 0.935 | 0.8 | 0.855 | 0.888 | 0.857 |
| relative | HistAvg | 0.951 | 0.96 | 0.949 | 0.594 | 0.548 | 0.542 |
| relative | LGB-clf[class_weight] | 0.948 | 0.953 | 0.977 | 0.869 | 0.773 | 0.702 |
| relative | LGB-clf[focal] | 0.947 | 0.963 | 0.979 | 0.866 | 0.789 | 0.705 |
| relative | LGB-clf[none] | 0.95 | 0.954 | 0.98 | 0.861 | 0.794 | 0.715 |
| relative | LGB-clf[undersample] | 0.944 | 0.942 | 0.976 | 0.857 | 0.78 | 0.704 |
| relative | LGB-reg/thr | 0.948 | 0.943 | 0.954 | 0.856 | 0.84 | 0.801 |
