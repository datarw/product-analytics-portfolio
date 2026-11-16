"""Проверки эксперимента: SRM, A/A placebo, разница пропорций.

Функции не зависят от датасета и тестируются на синтетике.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chisquare, mannwhitneyu, norm


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


def holm_adjust(p_values) -> np.ndarray:
    """Holm–Bonferroni: контроль FWER при нескольких метриках."""
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adjusted = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[idx]))
        adjusted[idx] = running
    return adjusted


def downsample_diff(df: pd.DataFrame, metric: str, rng: np.random.Generator,
                    n_iter: int = 1000, alpha: float = 0.05) -> dict:
    """SRM-чувствительность: много раз уменьшаем большую группу до размера меньшей.

    Это не исправляет смещение, но показывает, что эффект не создан лишними
    пользователями из перевешенной группы.
    """
    a = df.loc[df["version"] == "gate_30", metric].to_numpy(dtype=bool)
    b = df.loc[df["version"] == "gate_40", metric].to_numpy(dtype=bool)
    n = min(a.size, b.size)
    diffs = np.empty(n_iter)
    for i in range(n_iter):
        b_sample = rng.choice(b, size=n, replace=False)
        diffs[i] = b_sample.mean() - a.mean()
    return {
        "metric": metric,
        "n_per_arm": int(n),
        "n_iter": n_iter,
        "point_full": float(b.mean() - a.mean()),
        "mean_downsampled": float(diffs.mean()),
        "ci_low": float(np.percentile(diffs, 100 * alpha / 2)),
        "ci_high": float(np.percentile(diffs, 100 * (1 - alpha / 2))),
    }


def bootstrap_median_diff(a, b, rng: np.random.Generator,
                          n_iter: int = 1000, alpha: float = 0.05) -> dict:
    """Bootstrap CI for median(B) − median(A); heavy-tailed guardrail."""
    a = np.asarray(a)
    b = np.asarray(b)
    diffs = np.empty(n_iter)
    for i in range(n_iter):
        diffs[i] = (np.median(rng.choice(b, size=b.size, replace=True))
                    - np.median(rng.choice(a, size=a.size, replace=True)))
    return {
        "median_a": float(np.median(a)),
        "median_b": float(np.median(b)),
        "diff": float(np.median(b) - np.median(a)),
        "ci_low": float(np.percentile(diffs, 100 * alpha / 2)),
        "ci_high": float(np.percentile(diffs, 100 * (1 - alpha / 2))),
    }


def guardrail_table(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Retention + engagement guardrails с Holm-поправкой."""
    a = df[df["version"] == "gate_30"]
    b = df[df["version"] == "gate_40"]
    rows = []

    for metric in ("retention_1", "retention_7"):
        res = two_proportion_ztest(a[metric].sum(), len(a), b[metric].sum(), len(b))
        rows.append({"metric": metric, "kind": "retention", **res})

    zero_a = (a["sum_gamerounds"] == 0)
    zero_b = (b["sum_gamerounds"] == 0)
    res = two_proportion_ztest(zero_a.sum(), len(a), zero_b.sum(), len(b))
    rows.append({"metric": "zero_rounds", "kind": "guardrail", **res})

    mw = mannwhitneyu(a["sum_gamerounds"], b["sum_gamerounds"], alternative="two-sided")
    med = bootstrap_median_diff(a["sum_gamerounds"], b["sum_gamerounds"], rng)
    rows.append({
        "metric": "median_gamerounds", "kind": "guardrail",
        "p_a": med["median_a"], "p_b": med["median_b"],
        "diff": med["diff"], "ci_low": med["ci_low"], "ci_high": med["ci_high"],
        "z": np.nan, "p_value": float(mw.pvalue),
        "relative_lift": np.nan,
    })

    out = pd.DataFrame(rows)
    out["p_holm"] = holm_adjust(out["p_value"])
    return out


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
