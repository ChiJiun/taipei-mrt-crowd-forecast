"""Reproduce progress-report Table 2 (crowd warning, test 2026-09).

Crowd = next-hour (entries+exits) >= station P95 on training data. Decision threshold: highest
threshold with Recall >= 0.7 on validation (2026-08). Scores compared:
  HistAvg              hist_avg(total) / station threshold
  LightGBM-reg/thr     LightGBM-scaled regression of total(t+1), prediction / station threshold
  LightGBM-binary      LightGBM binary classifier on the same features

Usage: python src/40_crowd_baseline.py
Outputs results/clf_table2.csv, results/clf_table2.md, results/clf_pred_test_h1.parquet
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import RESULTS  # noqa: E402
from features import load_panel  # noqa: E402
from metrics import clf_metrics, pick_threshold  # noqa: E402

lib = importlib.import_module("40_crowd_lib")


def main():
    panel = load_panel()
    thr = lib.crowd_thresholds(panel)
    fr = lib.total_frame(panel, h=1, spatial=False)
    fr["crowd_thr"] = fr["station"].map(thr).astype(float)
    fr["label"] = (fr["y"] >= fr["crowd_thr"]).astype(int)
    tr, va, te = lib.split(fr)
    cols = lib.base_cols(fr)
    print(f"positive rate train {tr.label.mean():.4f} valid {va.label.mean():.4f} test {te.label.mean():.4f}")

    scores_va, scores_te = {}, {}
    scores_va["HistAvg"] = va.hist_avg / va.crowd_thr
    scores_te["HistAvg"] = te.hist_avg / te.crowd_thr
    m = lib.fit_scaled_reg(tr, va, cols)
    scores_va["LightGBM-reg/thr"] = lib.pred_scaled(m, va, cols) / va.crowd_thr
    scores_te["LightGBM-reg/thr"] = lib.pred_scaled(m, te, cols) / te.crowd_thr
    c = lib.fit_clf(tr, va, cols, mode="none")
    scores_va["LightGBM-binary"] = lib.pred_clf(c, va, cols)
    scores_te["LightGBM-binary"] = lib.pred_clf(c, te, cols)

    rows = []
    preds = te[["ts", "ts_target", "station", "y", "crowd_thr", "label"]].copy()
    for name in scores_te:
        t = pick_threshold(va.label, scores_va[name], 0.7)
        rows.append({"method": name, **clf_metrics(te.label, scores_te[name], t)})
        preds[name] = pd.Series(scores_te[name]).to_numpy()
    res = pd.DataFrame(rows)
    report = pd.DataFrame([{"method": "HistAvg (報告)", "AUC": 0.995, "AP": 0.937, "Precision": 0.949, "Recall": 0.578,
                            "F1": 0.718},
                           {"method": "LightGBM (報告)", "AUC": 0.997, "AP": 0.965, "Precision": 0.961, "Recall": 0.742,
                            "F1": 0.838}])
    out = pd.concat([res, report], ignore_index=True)
    out.to_csv(RESULTS / "clf_table2.csv", index=False)
    (RESULTS / "clf_table2.md").write_text(
        "# 表二重現：擁擠預警（測試集 2026-09，門檻＝驗證集 Recall≥0.7 的最高門檻）\n\n"
        f"正類率：訓練 {tr.label.mean():.3%}、驗證 {va.label.mean():.3%}、測試 {te.label.mean():.3%}\n\n"
        + lib.to_md(out) + "\n", encoding="utf-8")
    preds.to_parquet(RESULTS / "clf_pred_test_h1.parquet", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
