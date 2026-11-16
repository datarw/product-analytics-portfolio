"""Проверки эксперимента: SRM, A/A placebo, разница пропорций.

Функции не зависят от датасета и тестируются на синтетике.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chisquare, norm


def srm_test(counts, expected_ratio=None) -> dict:
    """Sample Ratio Mismatch: хи-квадрат наблюдаемых размеров групп против ожидаемых.

    `expected_ratio` по умолчанию 50/50. Возвращает p-value, доли и величину
    дисбаланса в процентных пунктах.
    """
    labels = list(counts.index) if isinstance(counts, pd.Series) else list(range(len(counts)))
    observed = np.asarray(counts, dtype=float)
    n = observed.sum()
    if expected_ratio is None:
        expected_ratio = np.ones(len(observed)) / len(observed)
    expected_ratio = np.asarray(expected_ratio, dtype=float)
    expected_ratio = expected_ratio / expected_ratio.sum()
    expected = n * expected_ratio
    chi2, p_value = chisquare(observed, expected)
    shares = observed / n
    return {
        "labels": labels,
        "observed": observed.tolist(),
        "expected": expected.tolist(),
        "shares": shares.tolist(),
        "chi2": float(chi2),
        "p_value": float(p_value),
        "max_imbalance_pp": float(np.max(np.abs(shares - expected_ratio)) * 100),
        "n": int(n),
    }


def two_proportion_ztest(success_a, n_a, success_b, n_b, alpha: float = 0.05) -> dict:
    """Разница пропорций B − A с z-тестом и нормальным доверительным интервалом."""
    p_a = success_a / n_a
    p_b = success_b / n_b
    diff = p_b - p_a
    p_pool = (success_a + success_b) / (n_a + n_b)
    se_pool = np.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
    z = diff / se_pool if se_pool > 0 else 0.0
    p_value = 2 * (1 - norm.cdf(abs(z))) if se_pool > 0 else 1.0

    se_unpooled = np.sqrt(p_a * (1 - p_a) / n_a + p_b * (1 - p_b) / n_b)
    z_crit = norm.ppf(1 - alpha / 2)
    ci_low = diff - z_crit * se_unpooled
    ci_high = diff + z_crit * se_unpooled
    return {
        "p_a": float(p_a),
        "p_b": float(p_b),
        "diff": float(diff),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "z": float(z),
        "p_value": float(p_value),
        "relative_lift": float(diff / p_a) if p_a > 0 else np.nan,
    }


def wilson_ci(success, n, alpha: float = 0.05):
    """Wilson interval for a proportion; better than normal at small n."""
    z = norm.ppf(1 - alpha / 2)
    p = success / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return float(center - half), float(center + half)


def aa_placebo(df: pd.DataFrame, metric: str, rng: np.random.Generator,
               n_iter: int = 1000, alpha: float = 0.05) -> dict:
    """A/A placebo: случайно делим пул 50/50 и считаем долю ложных срабатываний.

    Это проверка пайплайна, а не реального сплита: если процедура корректна,
    доля p < alpha близка к alpha.
    """
    values = df[metric].to_numpy(dtype=bool)
    n = len(values)
    p_values = np.empty(n_iter)
    for i in range(n_iter):
        mask = rng.random(n) < 0.5
        a, b = values[mask], values[~mask]
        p_values[i] = two_proportion_ztest(a.sum(), a.size, b.sum(), b.size, alpha)["p_value"]
    return {
        "metric": metric,
        "n_iter": n_iter,
        "alpha": alpha,
        "false_positive_rate": float((p_values < alpha).mean()),
        "p_values": p_values,
    }


def baseline_table(df: pd.DataFrame, metrics=("retention_1", "retention_7")) -> pd.DataFrame:
    """Описательная таблица по группам: доли, разница, CI, p-value."""
    rows = []
    for metric in metrics:
        a = df.loc[df["version"] == "gate_30", metric]
        b = df.loc[df["version"] == "gate_40", metric]
        res = two_proportion_ztest(a.sum(), len(a), b.sum(), len(b))
        rows.append({
            "metric": metric,
            "n_gate_30": len(a),
            "n_gate_40": len(b),
            "gate_30": float(a.mean()),
            "gate_40": float(b.mean()),
            **res,
        })
    return pd.DataFrame(rows)
