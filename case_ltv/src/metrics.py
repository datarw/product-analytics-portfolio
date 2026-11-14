"""Метрики ранжирования и калибровки для holdout-валидации LTV.

Две оси всегда вместе:
    ранжирование — Gini, Lorenz, decile capture/lift;
    калибровка   — MAE/RMSE и calibration ratio по срезам.
Bootstrap по клиентам возвращает интервал; сравнение без интервала не публикуется.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CALIBRATION_BAND = (0.8, 1.2)


def _order(y_score):
    return np.argsort(-np.asarray(y_score, dtype=float), kind="mergesort")


def gini(y_true, y_score) -> float:
    """Gini (Somers' D) ранжирования по убыванию прогноза."""
    y = np.asarray(y_true, dtype=float)
    if y.sum() <= 0:
        return np.nan
    y = y[_order(y_score)]
    cum = np.cumsum(y) / y.sum()
    x = np.arange(1, len(y) + 1) / len(y)
    area = np.trapezoid(np.concatenate([[0.0], cum]), np.concatenate([[0.0], x]))
    # сортировка от лучшего к худшему: кривая выше диагонали, Gini = 2A − 1
    return float(2 * area - 1)


def top_capture(y_true, y_score, frac: float = 0.1) -> float:
    """Доля фактической выручки в верхних frac клиентов по прогнозу."""
    y = np.asarray(y_true, dtype=float)
    if y.sum() <= 0:
        return np.nan
    k = max(1, int(round(frac * len(y))))
    return float(y[_order(y_score)][:k].sum() / y.sum())


def top_lift(y_true, y_score, frac: float = 0.1) -> float:
    """Decile gain: capture / доля клиентов."""
    return top_capture(y_true, y_score, frac) / frac


def decile_capture(y_true, y_score) -> np.ndarray:
    """Доля выручки по децилям от лучшего к худшему; сумма = 1."""
    y = np.asarray(y_true, dtype=float)
    if y.sum() <= 0:
        return np.full(10, np.nan)
    y = y[_order(y_score)]
    parts = np.array_split(y, 10)
    return np.array([p.sum() / y.sum() for p in parts])


def lorenz_points(y_true, y_score):
    """Точки кривой Лоренца: (доля клиентов, доля выручки)."""
    y = np.asarray(y_true, dtype=float)
    y = y[_order(y_score)]
    x = np.concatenate([[0.0], np.arange(1, len(y) + 1) / len(y)])
    cum = np.concatenate([[0.0], np.cumsum(y) / max(y.sum(), 1e-12)])
    return x, cum


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def calibration_ratio(y_true, y_pred) -> float:
    """mean(pred) / mean(actual); флаг вне [0.8, 1.2]."""
    actual = float(np.mean(y_true))
    return float(np.mean(y_pred) / actual) if actual > 0 else np.nan


def model_metrics(y_true, y_pred) -> dict:
    return {
        "gini": gini(y_true, y_pred),
        "top10_capture": top_capture(y_true, y_pred, 0.10),
        "top10_lift": top_lift(y_true, y_pred, 0.10),
        "top20_capture": top_capture(y_true, y_pred, 0.20),
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "calibration_ratio": calibration_ratio(y_true, y_pred),
    }


def bootstrap_ci(y_true, y_score, metric_fn, rng, n_boot: int = 1000,
                 alpha: float = 0.05) -> dict:
    """Percentile bootstrap по клиентам для одной метрики."""
    y = np.asarray(y_true, dtype=float)
    s = np.asarray(y_score, dtype=float)
    n = len(y)
    point = metric_fn(y, s)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        draws[b] = metric_fn(y[idx], s[idx])
    return {
        "point": float(point),
        "lo": float(np.nanpercentile(draws, 100 * alpha / 2)),
        "hi": float(np.nanpercentile(draws, 100 * (1 - alpha / 2))),
    }


def bootstrap_diff(y_true, score_a, score_b, metric_fn, rng,
                   n_boot: int = 1000, alpha: float = 0.05) -> dict:
    """Парный bootstrap разницы metric(a) − metric(b) на одних и тех же клиентах."""
    y = np.asarray(y_true, dtype=float)
    a = np.asarray(score_a, dtype=float)
    b = np.asarray(score_b, dtype=float)
    n = len(y)
    point = metric_fn(y, a) - metric_fn(y, b)
    draws = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        draws[i] = metric_fn(y[idx], a[idx]) - metric_fn(y[idx], b[idx])
    return {
        "point": float(point),
        "lo": float(np.nanpercentile(draws, 100 * alpha / 2)),
        "hi": float(np.nanpercentile(draws, 100 * (1 - alpha / 2))),
    }


def calibration_by(y_true, y_pred, groups) -> pd.DataFrame:
    """Calibration ratio по срезу: n, факт, прогноз, ratio, флаг."""
    df = pd.DataFrame({"actual": np.asarray(y_true, dtype=float),
                       "pred": np.asarray(y_pred, dtype=float),
                       "group": np.asarray(groups)})
    out = (df.groupby("group", observed=True)
           .agg(n=("actual", "size"), actual=("actual", "mean"), pred=("pred", "mean"))
           .reset_index())
    out["ratio"] = np.where(out["actual"] > 0, out["pred"] / out["actual"], np.nan)
    out["flag"] = ~out["ratio"].between(*CALIBRATION_BAND)
    return out.sort_values("n", ascending=False).reset_index(drop=True)


def coverage_at_top(y_true, y_score, frac: float):
    """Размер сегмента, его доля и покрытие выручки для верхних frac клиентов."""
    n = len(y_true)
    k = max(1, int(round(frac * n)))
    return {
        "k": k,
        "share_customers": k / n,
        "coverage": top_capture(y_true, y_score, k / n),
        "threshold": float(np.sort(np.asarray(y_score))[::-1][k - 1]),
    }


def smallest_segment_for_coverage(y_true, y_score, target: float = 0.60):
    """Минимальный верхний сегмент, покрывающий target выручки holdout."""
    y = np.asarray(y_true, dtype=float)
    s = np.asarray(y_score, dtype=float)
    order = _order(s)
    cum = np.cumsum(y[order]) / y.sum()
    k = int(np.searchsorted(cum, target) + 1)
    return {
        "k": k,
        "share_customers": k / len(y),
        "coverage": float(cum[k - 1]),
        "threshold": float(s[order][k - 1]),
    }


def shrink_predictions(pred, groups, n_obs, k: float = 10.0):
    """Shrinkage прогноза к среднему по группе: w = n/(n+k).

    Для сегмента B (одна покупка) n = 1, поэтому прогноз почти равен
    среднему группы «страна × бэнд первого чека»; для частых клиентов
    индивидуальный прогноз сохраняется.
    """
    pred = pd.Series(pred)
    groups = pd.Series(groups, index=pred.index)
    n_obs = pd.Series(n_obs, index=pred.index).astype(float)
    group_mean = pred.groupby(groups).transform("mean")
    w = n_obs / (n_obs + k)
    return w * pred + (1 - w) * group_mean
