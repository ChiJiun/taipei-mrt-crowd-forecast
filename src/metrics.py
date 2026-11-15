"""Shared evaluation helpers (regression + crowd classification)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from config import PEAK_HOURS, RAIN_MM  # noqa: E402


def peak_mask(frame: pd.DataFrame) -> np.ndarray:
    """Weekday (non-holiday) peak hours 7,8,17,18 of the *target* hour."""
    return (frame["hour"].isin(PEAK_HOURS) & (frame["is_holiday"] == 0)).to_numpy()


def reg_metrics(y, p, frame: pd.DataFrame | None = None) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    e = p - y
    res = {"MAE": np.abs(e).mean(), "RMSE": np.sqrt((e ** 2).mean())}
    if frame is not None:
        pk = peak_mask(frame)
        res["peak_MAE"] = np.abs(e[pk]).mean()
        if "precip" in frame:
            rain = (frame["precip"] >= RAIN_MM).to_numpy()
            if rain.any():
                res["rain_MAE"] = np.abs(e[rain]).mean()
                res["dry_MAE"] = np.abs(e[~rain]).mean()
    return {k: round(float(v), 2) for k, v in res.items()}


def pick_threshold(y_true, score, min_recall: float = 0.7) -> float:
    """Highest threshold whose recall >= min_recall (chosen on validation)."""
    y_true, score = np.asarray(y_true), np.asarray(score)
    order = np.sort(np.unique(score))[::-1]
    pos = y_true.sum()
    best = order[-1]
    # vectorised scan over candidate thresholds
    s_sorted = np.sort(score[y_true == 1])
    for t in np.quantile(score, np.linspace(1, 0, 2001)):
        rec = (s_sorted >= t).sum() / max(pos, 1)
        if rec >= min_recall:
            best = t
            break
    return float(best)


def clf_metrics(y_true, score, thr: float) -> dict:
    y_true, score = np.asarray(y_true), np.asarray(score)
    pred = score >= thr
    return {"AUC": round(roc_auc_score(y_true, score), 3), "AP": round(average_precision_score(y_true, score), 3),
            "Precision": round(precision_score(y_true, pred, zero_division=0), 3),
            "Recall": round(recall_score(y_true, pred), 3), "F1": round(f1_score(y_true, pred), 3),
            "thr": round(thr, 4), "pos_rate": round(y_true.mean(), 4)}
