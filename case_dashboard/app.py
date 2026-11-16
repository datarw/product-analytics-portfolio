"""Streamlit-дашборд юнит-экономики.

Запуск из корня репозитория:
    streamlit run case_dashboard/app.py

Данные — витрина `../data/processed/dashboard_unit_economics.parquet` (вне git).
Каналы синтетические; CAC и margin — сценарные допущения.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "case_dashboard" / "src"))
from paths import PROCESSED  # noqa: E402
from unit_economics import (add_unit_economics, channel_summary,  # noqa: E402
                            cohort_summary, reallocate_budget, scenario_table)

st.set_page_config(page_title="Unit economics", layout="wide")
st.title("Юнит-экономика каналов")
st.caption("Синтетические каналы; CAC и margin — сценарные допущения. Данные вне git.")


@st.cache_data
def load_data():
    df = pd.read_parquet(PROCESSED / "dashboard_unit_economics.parquet").set_index("customer_id")
    meta = json.loads((PROCESSED / "dashboard_meta.json").read_text(encoding="utf-8"))
    return df, meta


base, meta = load_data()

with st.sidebar:
    st.header("Фильтры")
    scenario = st.selectbox("Сценарий", ["base", "optimistic", "pessimistic"], index=0)
    channels = st.multiselect("Каналы", sorted(base.channel.unique()), default=sorted(base.channel.unique()))
    cohorts = st.multiselect("Когорты", sorted(base.cohort.unique()), default=sorted(base.cohort.unique()))
    st.divider()
    st.caption("Пороги: pause — LTV:CAC < 1 или payback > 12; scale — LTV:CAC ≥ 3 и payback ≤ 6.")

view = base[base.channel.isin(channels) & base.cohort.isin(cohorts)].copy()
view = add_unit_economics(view.drop(columns=[c for c in
                                             ["ltv_gp", "cac", "contribution", "ltv_cac",
                                              "payback_months", "decision"] if c in view],
                                errors="ignore"), scenario)
ch = channel_summary(view)
scen = scenario_table(view)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Contribution margin", f"£{view.contribution.sum():,.0f}")
c2.metric("LTV:CAC", f"{view.ltv_gp.sum() / view.cac.sum():.2f}" if view.cac.sum() > 0 else "n/a")
c3.metric("Payback", f"{view.cac.sum() / (view.ltv_gp.sum() / 12):.1f} мес" if view.ltv_gp.sum() > 0 else "n/a")
c4.metric("Acquisition budget", f"£{view.cac.sum():,.0f}")

st.subheader("Каналы: LTV:CAC и решение")
left, right = st.columns([1.2, 1])
with left:
    st.bar_chart(ch.set_index("channel")["ltv_cac"], height=320)
with right:
    st.dataframe(
        ch[["channel", "n", "budget", "ltv_cac", "payback_months", "decision"]]
        .style.format({"budget": "£{:,.0f}", "ltv_cac": "{:.2f}", "payback_months": "{:.1f}"}),
        use_container_width=True, hide_index=True,
    )

st.subheader("Сценарии")
st.dataframe(
    scen[["scenario", "contribution", "ltv_cac", "payback_months", "pause_channels", "scale_channels"]]
    .style.format({"contribution": "£{:,.0f}", "ltv_cac": "{:.2f}", "payback_months": "{:.1f}"}),
    use_container_width=True, hide_index=True,
)

st.subheader("Перераспределение бюджета")
s1, s2 = st.columns(2)
with s1:
    shift = st.slider("Доля бюджета keep-каналов", 0.0, 0.5, 0.2, 0.05)
with s2:
    elasticity = st.slider("Эластичность CAC при масштабировании", 0.0, 1.5, 0.5, 0.1)
realloc, summary = reallocate_budget(view, shift=shift, elasticity=elasticity)
if realloc.empty:
    st.info("Нет одновременно keep- и scale-каналов для перераспределения.")
else:
    st.metric("Uplift contribution", f"£{summary['uplift']:,.0f}",
              f"{summary['uplift_pct']:.1%} к текущему")
    st.dataframe(
        realloc[["channel", "budget_old", "budget_new", "cac_old", "cac_new",
                 "customers_new", "contribution_new"]]
        .style.format({"budget_old": "£{:,.0f}", "budget_new": "£{:,.0f}",
                       "cac_old": "£{:,.0f}", "cac_new": "£{:,.0f}",
                       "customers_new": "{:,.0f}", "contribution_new": "£{:,.0f}"}),
        use_container_width=True, hide_index=True,
    )

st.subheader("Что делать")
pause = ch[ch.decision == "pause"]["channel"].tolist()
scale = ch[ch.decision == "scale"]["channel"].tolist()
keep = ch[ch.decision == "keep"]["channel"].tolist()
st.markdown(
    f"- **Pause:** {', '.join(pause) if pause else 'нет каналов ниже порога'}.\n"
    f"- **Scale:** {', '.join(scale) if scale else 'нет'}.\n"
    f"- **Keep:** {', '.join(keep) if keep else 'нет'}.\n"
    "- Рост бюджета на scale-каналы поднимает CAC; смотрите uplift при эластичности, "
    "а не только базовый LTV:CAC."
)
