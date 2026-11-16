"""Юнит-экономика каналов: синтетические каналы, CAC-сценарии, payback.

Каналы синтезированы на уровне клиента: это сценарная надстройка, а не
наблюдаемые данные. CAC по каналам — допущение; gross margin 30 % — допущение
из LTV-кейса. Все функции принимают RNG явно.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CHANNELS = ["paid_search", "social", "email", "referral", "organic"]

# Сценарные CAC на привлечённого клиента, £. Не данные.
CHANNEL_CAC = {
    "paid_search": 320.0,
    "social": 200.0,
    "email": 30.0,
    "referral": 60.0,
    "organic": 20.0,
}

GROSS_MARGIN = 0.30
# LTV-кейс: calibration ratio 1.364; обратная величина — консервативная поправка.
LTV_CALIBRATION = 1 / 1.364

SCENARIOS = {
    "optimistic": {"ltv_factor": 1.0, "cac_factor": 0.85},
    "base": {"ltv_factor": LTV_CALIBRATION, "cac_factor": 1.0},
    "pessimistic": {"ltv_factor": 0.9 * LTV_CALIBRATION, "cac_factor": 1.25},
}

PAUSE_LTV_CAC = 1.0
SCALE_LTV_CAC = 3.0
PAUSE_PAYBACK = 12.0
SCALE_PAYBACK = 6.0


def assign_channels(df: pd.DataFrame, rng: np.random.Generator) -> pd.Series:
    """Синтетический канал: softmax по первому чеку.

    Дорогие первые чеки чаще приходят из email/referral, дешёвые — из
    paid_search/social. Это сценарная связь LTV и канала, не наблюдение.
    """
    z = (df["first_check"] - df["first_check"].mean()) / df["first_check"].std()
    base = np.array([0.2, 0.1, 0.1, -0.2, -0.1])
    slope = np.array([-0.4, -0.5, 0.3, 0.5, 0.1])
    logits = base[None, :] + slope[None, :] * z.to_numpy()[:, None]
    p = np.exp(logits - logits.max(axis=1, keepdims=True))
    p = p / p.sum(axis=1, keepdims=True)
    u = rng.random(len(df))[:, None]
    idx = (u > np.cumsum(p, axis=1)).sum(axis=1)
    return pd.Series(np.array(CHANNELS)[idx], index=df.index, name="channel")


def add_unit_economics(df: pd.DataFrame, scenario: str = "base") -> pd.DataFrame:
    """LTV gross profit, CAC, contribution, LTV:CAC и payback."""
    cfg = SCENARIOS[scenario]
    out = df.copy()
    out["ltv_gp"] = out["pv"] * cfg["ltv_factor"] * GROSS_MARGIN
    out["cac"] = out["channel"].map(CHANNEL_CAC) * cfg["cac_factor"]
    out["contribution"] = out["ltv_gp"] - out["cac"]
    out["ltv_cac"] = np.where(out["cac"] > 0, out["ltv_gp"] / out["cac"], np.inf)
    out["payback_months"] = np.where(out["ltv_gp"] > 0, out["cac"] / (out["ltv_gp"] / 12), np.inf)
    out["decision"] = np.select(
        [(out["ltv_cac"] < PAUSE_LTV_CAC) | (out["payback_months"] > PAUSE_PAYBACK),
         (out["ltv_cac"] >= SCALE_LTV_CAC) & (out["payback_months"] <= SCALE_PAYBACK)],
        ["pause", "scale"], default="keep",
    )
    return out


def channel_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Агрегат по каналу: бюджет, LTV gross profit, contribution, ratio, payback."""
    g = (df.groupby("channel")
         .agg(n=("ltv_gp", "size"), budget=("cac", "sum"),
              ltv_gp=("ltv_gp", "sum"), contribution=("contribution", "sum"),
              ltv_gp_mean=("ltv_gp", "mean"), cac_mean=("cac", "mean"))
         .reset_index())
    g["ltv_cac"] = np.where(g["budget"] > 0, g["ltv_gp"] / g["budget"], np.inf)
    g["payback_months"] = np.where(g["ltv_gp"] > 0, g["budget"] / (g["ltv_gp"] / 12), np.inf)
    g["decision"] = np.select(
        [(g["ltv_cac"] < PAUSE_LTV_CAC) | (g["payback_months"] > PAUSE_PAYBACK),
         (g["ltv_cac"] >= SCALE_LTV_CAC) & (g["payback_months"] <= SCALE_PAYBACK)],
        ["pause", "scale"], default="keep",
    )
    return g.sort_values("ltv_cac", ascending=False).reset_index(drop=True)


def cohort_summary(df: pd.DataFrame) -> pd.DataFrame:
    g = (df.groupby(["cohort", "channel"])
         .agg(n=("ltv_gp", "size"), budget=("cac", "sum"),
              ltv_gp=("ltv_gp", "sum"), contribution=("contribution", "sum"))
         .reset_index())
    g["ltv_cac"] = np.where(g["budget"] > 0, g["ltv_gp"] / g["budget"], np.inf)
    g["payback_months"] = np.where(g["ltv_gp"] > 0, g["budget"] / (g["ltv_gp"] / 12), np.inf)
    return g


def scenario_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scenario in SCENARIOS:
        s = add_unit_economics(df.drop(columns=[c for c in
                                               ["ltv_gp", "cac", "contribution", "ltv_cac",
                                                "payback_months", "decision"] if c in df],
                                   errors="ignore"), scenario)
        budget = s["cac"].sum()
        ltv = s["ltv_gp"].sum()
        contribution = s["contribution"].sum()
        ch = channel_summary(s)
        rows.append({
            "scenario": scenario,
            "ltv_gp": ltv,
            "budget": budget,
            "contribution": contribution,
            "ltv_cac": ltv / budget if budget > 0 else np.nan,
            "payback_months": budget / (ltv / 12) if ltv > 0 else np.nan,
            "pause_channels": int((ch["decision"] == "pause").sum()),
            "scale_channels": int((ch["decision"] == "scale").sum()),
            "pause_customers": int((s["decision"] == "pause").sum()),
            "scale_customers": int((s["decision"] == "scale").sum()),
        })
    return pd.DataFrame(rows)


def reallocate_budget(df: pd.DataFrame, shift: float = 0.2,
                      elasticity: float = 0.5) -> tuple[pd.DataFrame, dict]:
    """Перекладывает shift доли бюджета keep-каналов в scale-каналы.

    CAC получающих каналов растёт: `cac_new = cac*(1+elasticity*(budget_new/budget-1))`.
    LTV на клиента считается постоянной; это сценарное допущение, не прогноз.
    """
    ch = channel_summary(df).set_index("channel")
    senders = ch[ch["decision"] == "keep"]
    # organic не масштабируется бюджетом: его нельзя «купить» больше.
    receivers = ch[(ch["decision"] == "scale") & (ch.index != "organic")]
    if senders.empty or receivers.empty:
        return pd.DataFrame(), {"uplift": 0.0, "reason": "no senders or receivers"}

    send_budget = senders["budget"].sum()
    shift_amount = shift * send_budget
    recv_weight = receivers["budget"] / receivers["budget"].sum()

    rows = []
    for channel, row in ch.iterrows():
        budget_old = float(row["budget"])
        cac_old = float(row["cac_mean"])
        if channel in senders.index:
            budget_new = budget_old - shift_amount * (budget_old / send_budget)
            cac_new = cac_old
        elif channel in receivers.index:
            budget_new = budget_old + shift_amount * float(recv_weight[channel])
            cac_new = cac_old * (1 + elasticity * (budget_new / budget_old - 1))
        else:
            budget_new = budget_old
            cac_new = cac_old
        customers_new = budget_new / cac_new if cac_new > 0 else 0.0
        contribution_new = customers_new * (float(row["ltv_gp_mean"]) - cac_new)
        rows.append({
            "channel": channel,
            "budget_old": budget_old,
            "budget_new": budget_new,
            "cac_old": cac_old,
            "cac_new": cac_new,
            "customers_old": float(row["n"]),
            "customers_new": customers_new,
            "contribution_old": float(row["contribution"]),
            "contribution_new": contribution_new,
        })
    out = pd.DataFrame(rows)
    base = out["contribution_old"].sum()
    new = out["contribution_new"].sum()
    summary = {
        "shift": shift,
        "elasticity": elasticity,
        "contribution_old": float(base),
        "contribution_new": float(new),
        "uplift": float(new - base),
        "uplift_pct": float((new - base) / base) if base != 0 else np.nan,
        "shifted_budget": float(shift_amount),
    }
    return out, summary


def reallocation_grid(df: pd.DataFrame, shifts=(0.0, 0.1, 0.2, 0.3),
                      elasticities=(0.0, 0.25, 0.5, 1.0)) -> pd.DataFrame:
    rows = []
    for s in shifts:
        for e in elasticities:
            _, summary = reallocate_budget(df, shift=s, elasticity=e)
            rows.append({"shift": s, "elasticity": e, "uplift": summary.get("uplift", 0.0),
                         "uplift_pct": summary.get("uplift_pct", 0.0)})
    return pd.DataFrame(rows)
