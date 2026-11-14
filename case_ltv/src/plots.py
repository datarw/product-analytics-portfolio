"""Графики кейса LTV. Все функции сохраняют PNG и возвращают путь."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from metrics import lorenz_points


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    return path


def plot_split_timeline(tx, cutoff, holdout_end, path):
    fig, ax = plt.subplots(figsize=(10, 2.8))
    start, end = tx["date"].min(), tx["date"].max()
    ax.axvspan(start, cutoff, color="#4C78A8", alpha=.25, label="calibration")
    ax.axvspan(cutoff, holdout_end, color="#F58518", alpha=.25, label="holdout")
    ax.axvspan(holdout_end, end, color="#BAB0AC", alpha=.35, label="excluded (partial)")
    ax.axvline(cutoff, color="black", lw=1)
    ax.axvline(holdout_end, color="black", lw=1, ls="--")
    ax.set(yticks=[], xlabel="дата", title="Calibration / holdout split")
    ax.legend(loc="upper left", frameon=False, ncol=3)
    return _save(fig, path)


def plot_lorenz(y_true, predictions: dict, path, title="Кривая Лоренца на holdout"):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="равенство")
    for name, pred in predictions.items():
        x, cum = lorenz_points(y_true, pred)
        ax.plot(x, cum, lw=1.8, label=name)
    ax.set(xlabel="доля клиентов (по убыванию прогноза)",
           ylabel="доля фактической выручки holdout", title=title)
    ax.legend(frameon=False)
    return _save(fig, path)


def plot_model_comparison(table: pd.DataFrame, path,
                          title="Модель × метрика на holdout"):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    t = table.sort_values("gini")
    axes[0].barh(t["model"], t["gini"], color="#4C78A8")
    axes[0].set(xlabel="Gini", title="Ранжирование")
    t2 = table.sort_values("mae")
    axes[1].barh(t2["model"], t2["mae"], color="#F58518")
    axes[1].set(xlabel="MAE, £ на клиента", title="Калибровка уровня")
    return _save(fig, path)


def plot_calibration_by_decile(y_true, y_pred, path,
                               title="Калибровка по децилям прогноза"):
    df = pd.DataFrame({"actual": y_true, "pred": y_pred})
    df["decile"] = pd.qcut(df["pred"].rank(method="first"), 10, labels=False) + 1
    g = df.groupby("decile").agg(actual=("actual", "mean"), pred=("pred", "mean"))
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(g.index, g["actual"], "o-", label="факт holdout")
    ax.plot(g.index, g["pred"], "s--", label="прогноз")
    ax.set(xlabel="дециль прогноза (1 = худший, 10 = лучший)", ylabel="средняя выручка, £",
           title=title)
    ax.legend(frameon=False)
    return _save(fig, path)


def plot_horizon(curve: pd.DataFrame, path, chosen: int = 12):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(curve["horizon"], curve["predicted"], "o-", label="прогноз BG/NBD")
    obs = curve.dropna(subset=["observed"])
    ax.plot(obs["horizon"], obs["observed"], "s-", label="факт (где наблюдается)")
    ax.axvline(chosen, color="black", lw=1, ls="--", label=f"выбранный горизонт {chosen} мес")
    ax.set(xlabel="горизонт, месяцев", ylabel="net-выручка на клиента, £",
           title="Ответ vs горизонт")
    ax.legend(frameon=False)
    return _save(fig, path)


def plot_segments(seg: pd.DataFrame, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(seg["segment"], seg["customers_share"], color="#4C78A8")
    axes[0].set(ylabel="доля клиентов", title="Размер сегментов")
    axes[1].bar(seg["segment"], seg["revenue_share"], color="#F58518")
    axes[1].set(ylabel="доля выручки holdout", title="Вклад в выручку")
    return _save(fig, path)


def plot_shrinkage(before, after, y_true, path,
                   title="Shrinkage сегмента B: до и после"):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.scatter(before, y_true, s=8, alpha=.25, label="до shrinkage")
    ax.scatter(after, y_true, s=8, alpha=.25, label="после shrinkage")
    lim = max(float(np.nanmax(before)), float(np.nanmax(after))) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=1, label="y = x")
    ax.set(xlabel="прогноз, £", ylabel="факт holdout, £", title=title)
    ax.legend(frameon=False)
    return _save(fig, path)


def plot_posterior(summary: pd.DataFrame, path,
                   title="BG/NBD: постериор параметров"):
    params = [p for p in ("r", "alpha", "a", "b") if p in summary.index]
    fig, axes = plt.subplots(1, len(params), figsize=(3.2 * len(params), 3.2))
    if len(params) == 1:
        axes = [axes]
    for ax, p in zip(axes, params):
        row = summary.loc[p]
        ax.errorbar([0], [row["mean"]],
                    yerr=[[row["mean"] - row["eti89_lb"]], [row["eti89_ub"] - row["mean"]]],
                    fmt="o", capsize=6)
        ax.set(xticks=[], ylabel=p, title=f"r_hat={row['r_hat']:.3f}")
    fig.suptitle(title, y=1.02)
    return _save(fig, path)
