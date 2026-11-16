"""Статические скриншоты дашборда и HTML-превью.

Streamlit-приложение — `case_dashboard/app.py`; эти картинки нужны, если
публичная ссылка умрёт. Запуск:
    python case_dashboard/src/make_screenshots.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "case_dashboard" / "src"))
from unit_economics import channel_summary, scenario_table  # noqa: E402

from paths import CASE_DASHBOARD, PROCESSED  # noqa: E402

FIG = CASE_DASHBOARD / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
REPORTS = CASE_DASHBOARD / "reports"

COLORS = {"pause": "#E45756", "keep": "#BAB0AC", "scale": "#54A24B"}


def load():
    df = pd.read_parquet(PROCESSED / "dashboard_unit_economics.parquet").set_index("customer_id")
    return df, channel_summary(df), scenario_table(df)


def fig_overview(df, ch, scen):
    fig = plt.figure(figsize=(12, 7), dpi=150)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.2])
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.axis("off")
    contribution = df.contribution.sum()
    ltv_cac = df.ltv_gp.sum() / df.cac.sum()
    payback = df.cac.sum() / (df.ltv_gp.sum() / 12)
    budget = df.cac.sum()
    ax0.text(0, 1.0, "Unit economics", fontsize=16, weight="bold", va="top")
    ax0.text(0, 0.78, f"Contribution: £{contribution:,.0f}", fontsize=12)
    ax0.text(0, 0.62, f"LTV:CAC: {ltv_cac:.2f}", fontsize=12)
    ax0.text(0, 0.46, f"Payback: {payback:.1f} мес", fontsize=12)
    ax0.text(0, 0.30, f"Acquisition budget: £{budget:,.0f}", fontsize=12)
    ax0.text(0, 0.10, "Каналы синтетические; CAC и margin — допущения.", fontsize=9, color="#666666")

    ax1 = fig.add_subplot(gs[0, 1])
    t = ch.sort_values("ltv_cac")
    ax1.barh(t.channel, t.ltv_cac, color=[COLORS[d] for d in t.decision])
    ax1.axvline(1, color="black", ls="--", lw=1)
    ax1.axvline(3, color="black", ls=":", lw=1)
    ax1.set(xlabel="LTV:CAC", title="Каналы: LTV:CAC")

    ax2 = fig.add_subplot(gs[1, 0])
    coh = df.groupby("cohort").contribution.sum()
    ax2.plot(range(len(coh)), coh.values, "o-", color="#4C78A8")
    ax2.set_xticks(range(len(coh)), coh.index, rotation=45, ha="right")
    ax2.set(ylabel="contribution, £", title="Contribution по когортам")

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.bar(scen.scenario, scen.contribution, color=["#F58518", "#4C78A8", "#E45756"])
    ax3.set(ylabel="contribution, £", title="Сценарии")
    fig.tight_layout()
    fig.savefig(FIG / "01-dashboard-overview.png")
    plt.close(fig)


def fig_channels(ch):
    t = ch.sort_values("ltv_cac")
    fig, ax = plt.subplots(figsize=(9, 4.2), dpi=150)
    bars = ax.barh(t.channel, t.ltv_cac, color=[COLORS[d] for d in t.decision])
    ax.axvline(1, color="black", ls="--", lw=1, label="pause < 1")
    ax.axvline(3, color="black", ls=":", lw=1, label="scale ≥ 3")
    for bar, (_, row) in zip(bars, t.iterrows()):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f"payback {row.payback_months:.1f} мес", va="center", fontsize=8)
    ax.set(xlabel="LTV:CAC", title="Каналы: LTV:CAC, payback и решение")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "02-channel-unit-economics.png")
    plt.close(fig)


def fig_cohorts(df):
    pivot = (df.groupby(["cohort", "channel"]).contribution.sum()
             .unstack(fill_value=0))
    fig, ax = plt.subplots(figsize=(10, 4.4), dpi=150)
    ax.stackplot(range(len(pivot)), [pivot[c].values for c in pivot.columns],
                 labels=pivot.columns)
    ax.set_xticks(range(len(pivot)), pivot.index, rotation=45, ha="right")
    ax.set(ylabel="contribution, £", title="Contribution по когортам и каналам")
    ax.legend(frameon=False, ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "03-cohort-contribution.png")
    plt.close(fig)


def fig_scenarios(scen):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=150)
    axes[0].bar(scen.scenario, scen.contribution, color=["#F58518", "#4C78A8", "#E45756"])
    axes[0].set(ylabel="contribution, £", title="Contribution по сценариям")
    axes[1].bar(scen.scenario, scen.ltv_cac, color=["#F58518", "#4C78A8", "#E45756"])
    axes[1].axhline(1, color="black", ls="--", lw=1)
    axes[1].axhline(3, color="black", ls=":", lw=1)
    axes[1].set(ylabel="LTV:CAC", title="LTV:CAC по сценариям")
    fig.tight_layout()
    fig.savefig(FIG / "04-scenarios.png")
    plt.close(fig)


def fig_reallocation(grid):
    pivot = grid.pivot(index="shift", columns="elasticity", values="uplift_pct") * 100
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
    im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)), [f"{c:.2f}" for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)), [f"{i:.0%}" for i in pivot.index])
    ax.set(xlabel="эластичность CAC", ylabel="доля перекладываемого бюджета",
           title="Uplift contribution при перераспределении, %")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.values[i, j]:.1f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="uplift, %")
    fig.tight_layout()
    fig.savefig(FIG / "05-reallocation.png")
    plt.close(fig)


def write_html(df, ch, scen):
    rows = "".join(
        f"<tr><td>{r.channel}</td><td>{r.n}</td><td>£{r.budget:,.0f}</td>"
        f"<td>{r.ltv_cac:.2f}</td><td>{r.payback_months:.1f}</td><td>{r.decision}</td></tr>"
        for r in ch.itertuples()
    )
    scen_rows = "".join(
        f"<tr><td>{r.scenario}</td><td>£{r.contribution:,.0f}</td>"
        f"<td>{r.ltv_cac:.2f}</td><td>{r.payback_months:.1f}</td></tr>"
        for r in scen.itertuples()
    )
    html = f"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><title>Unit economics dashboard</title>
<style>body{{font-family:Arial,sans-serif;max-width:1100px;margin:32px auto;color:#222}}
img{{max-width:100%;border:1px solid #ddd;margin:12px 0}}table{{border-collapse:collapse;margin:12px 0}}
td,th{{border:1px solid #ddd;padding:6px 10px;text-align:right}}th{{background:#f4f4f4}}
.small{{color:#666;font-size:13px}}</style></head><body>
<h1>Юнит-экономика каналов</h1>
<p class="small">Синтетические каналы; CAC и margin — сценарные допущения. Данные вне git.</p>
<img src="figures/01-dashboard-overview.png" alt="overview">
<h2>Каналы</h2><table><tr><th>channel</th><th>n</th><th>budget</th><th>LTV:CAC</th><th>payback</th><th>decision</th></tr>{rows}</table>
<h2>Сценарии</h2><table><tr><th>scenario</th><th>contribution</th><th>LTV:CAC</th><th>payback</th></tr>{scen_rows}</table>
<img src="figures/02-channel-unit-economics.png" alt="channels">
<img src="figures/03-cohort-contribution.png" alt="cohorts">
<img src="figures/04-scenarios.png" alt="scenarios">
<img src="figures/05-reallocation.png" alt="reallocation">
</body></html>"""
    (REPORTS / "dashboard.html").write_text(html, encoding="utf-8")


if __name__ == "__main__":
    df, ch, scen = load()
    fig_overview(df, ch, scen)
    fig_channels(ch)
    fig_cohorts(df)
    fig_scenarios(scen)
    grid = pd.read_csv(PROCESSED / "dashboard_reallocation_grid.csv")
    fig_reallocation(grid)
    write_html(df, ch, scen)
    print("written", FIG)
