"""Собирает витрину дашборда: каналы, unit economics, SQLite и CSV.

Запуск из корня репозитория:
    python case_dashboard/src/build_mart.py
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from unit_economics import (
    CHANNEL_CAC,
    GROSS_MARGIN,
    SCENARIOS,
    add_unit_economics,
    assign_channels,
    channel_summary,
    cohort_summary,
    reallocate_budget,
    reallocation_grid,
    scenario_table,
)

from paths import CASE_DASHBOARD, PROCESSED, RANDOM_SEED  # noqa: E402

SQL = CASE_DASHBOARD / "sql" / "mart.sql"
DB = PROCESSED / "dashboard.sqlite"
MART = PROCESSED / "dashboard_unit_economics.parquet"


def load_base() -> pd.DataFrame:
    split = pd.read_parquet(PROCESSED / "ltv_split.parquet")
    disc = pd.read_parquet(PROCESSED / "discounted.parquet")
    spend = pd.read_parquet(PROCESSED / "spend.parquet")
    df = split.join(disc[["pv"]]).join(spend[["first_check"]])
    # 2009-12 — лево-цензурированная смесь, не когорта привлечения.
    df = df[df["cohort"] != "2009-12"].copy()
    rng = np.random.default_rng(RANDOM_SEED)
    df["channel"] = assign_channels(df, rng)
    return df


def build() -> pd.DataFrame:
    df = load_base()
    base = add_unit_economics(df, "base")

    base.reset_index().to_parquet(MART, index=False)

    with sqlite3.connect(DB) as con:
        base.reset_index().to_sql("unit_economics", con, if_exists="replace", index=False)
        con.executescript(SQL.read_text(encoding="utf-8"))

    ch = channel_summary(base)
    coh = cohort_summary(base)
    scen = scenario_table(base)
    realloc, realloc_sum = reallocate_budget(base, shift=0.2, elasticity=0.5)
    grid = reallocation_grid(base)

    ch.to_csv(PROCESSED / "dashboard_channel_summary.csv", index=False)
    coh.to_csv(PROCESSED / "dashboard_cohort_summary.csv", index=False)
    scen.to_csv(PROCESSED / "dashboard_scenarios.csv", index=False)
    realloc.to_csv(PROCESSED / "dashboard_reallocation.csv", index=False)
    grid.to_csv(PROCESSED / "dashboard_reallocation_grid.csv", index=False)

    scan = reallocation_grid(base, shifts=(0.2,), elasticities=tuple(np.linspace(0, 2, 41)))
    crossing = scan[scan["uplift"] <= 0]
    break_even = float(crossing["elasticity"].iloc[0]) if not crossing.empty else None

    meta = {
        "gross_margin": GROSS_MARGIN,
        "ltv_calibration": 1 / 1.364,
        "channel_cac": CHANNEL_CAC,
        "scenarios": SCENARIOS,
        "thresholds": {"pause_ltv_cac": 1.0, "pause_payback_months": 12,
                       "scale_ltv_cac": 3.0, "scale_payback_months": 6},
        "reallocation": realloc_sum,
        "break_even_elasticity_20pct_shift": break_even,
    }
    (PROCESSED / "dashboard_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return base, ch, scen, realloc_sum


if __name__ == "__main__":
    base, ch, scen, realloc_sum = build()
    print("unit economics rows:", len(base))
    print(ch[["channel", "n", "budget", "ltv_cac", "payback_months", "decision"]].round(3).to_string(index=False))
    print(scen.round(3).to_string(index=False))
    print("reallocation 20% shift / 0.5 elasticity:", realloc_sum)
