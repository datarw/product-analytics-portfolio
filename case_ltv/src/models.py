"""Вероятностные LTV-модели: BG/NBD, MBG/NBD, Pareto/NBD, Gamma-Gamma.

Все формулы реализованы явно и проверяются ассертами в ноутбуках (§5а).
Параметры — MLE, кроме основного BG/NBD: он фитится байесовски в ноутбуке 06,
а здесь лежат детерминированные формулы для прогноза и сравнения.

Конвенция (журнал, 2026-09-13):
    x       frequency = число повторных дней с покупкой (дней минус 1)
    t_x     recency = дней между первой и последней покупкой
    T       дней от первой покупки до cutoff
    m       средняя net-выручка повторного дня с покупкой
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln, logsumexp

DAYS_PER_MONTH = 30


# ---------------------------------------------------------------- BG/NBD

def bgnbd_loglik(params, x, t_x, T):
    """Индивидуальное лог-правдоподобие BG/NBD (Fader–Hardie–Lee, 2005).

    L = Γ(r+x)/Γ(r) α^r * C * [ a/(b+x-1) /(α+t_x)^(r+x) + 1/(α+T)^(r+x) ],
    C = Γ(a+b)Γ(b+x) / [Γ(a+b+x)Γ(b)].

    Множитель t_x^(x-1)/(x-1)! опущен: он не зависит от параметров и не
    влияет на MLE. Для x = 0 правдоподобие равно (α/(α+T))^r.
    """
    r, alpha, a, b = params
    x = np.asarray(x, dtype=float)
    t_x = np.asarray(t_x, dtype=float)
    T = np.asarray(T, dtype=float)
    out = np.empty_like(x)

    m0 = x == 0
    out[m0] = r * (np.log(alpha) - np.log(alpha + T[m0]))

    m = ~m0
    xm, txm, Tm = x[m], t_x[m], T[m]
    log_common = (gammaln(r + xm) - gammaln(r) + r * np.log(alpha)
                  + gammaln(a + b) + gammaln(b + xm)
                  - gammaln(a + b + xm) - gammaln(b))
    log_term1 = np.log(a / (b + xm - 1)) - (r + xm) * np.log(alpha + txm)
    log_term2 = -(r + xm) * np.log(alpha + Tm)
    out[m] = log_common + np.logaddexp(log_term1, log_term2)
    return out


def bgnbd_expected_purchases(params, x, t_x, T, t):
    """E[число покупок в (T, T+t] | история].

    Это ожидание уже проинтегрировано по «жив/мёртв»: множитель p(alive)
    сюда не входит и повторно применяться не должен (§5а-4).
    """
    r, alpha, a, b = params
    x = np.asarray(x, dtype=float)
    t_x = np.asarray(t_x, dtype=float)
    T = np.asarray(T, dtype=float)
    log_l = bgnbd_loglik(params, x, t_x, T)
    log_num = (gammaln(r + x + 1) - gammaln(r) + r * np.log(alpha)
               + gammaln(a + b) + gammaln(b + x) - gammaln(b) - gammaln(a + b + x)
               - (r + x + 1) * np.log(alpha + T))
    return t * np.exp(log_num - log_l)


# ---------------------------------------------------------------- MBG/NBD

def mbg_nbd_loglik(params, x, t_x, T):
    """Лог-правдоподобие MBG/NBD (Batislam et al., 2007; Wagner & Hoppe, 2008).

    Отличие от BG/NBD: клиент может отвалиться в момент 0, поэтому для x = 0
    появляется множитель b/(a+b), а для x > 0 — другой вес слагаемых.
    """
    r, alpha, a, b = params
    x = np.asarray(x, dtype=float)
    t_x = np.asarray(t_x, dtype=float)
    T = np.asarray(T, dtype=float)

    A1 = gammaln(r + x) - gammaln(r) + r * np.log(alpha)
    A2 = (gammaln(a + b) + gammaln(b + x + 1)
          - gammaln(b) - gammaln(a + b + x + 1))
    A3 = -(r + x) * np.log(alpha + T)
    A4 = np.where(x > 0, np.log(a / (b + x)) - (r + x) * np.log(alpha + t_x), -np.inf)
    return A1 + A2 + np.logaddexp(A3, A4)


def mbg_nbd_expected_purchases(params, x, t_x, T, t):
    """E[покупок в (T, T+t]] для MBG/NBD; тоже без повторного p(alive)."""
    r, alpha, a, b = params
    x = np.asarray(x, dtype=float)
    t_x = np.asarray(t_x, dtype=float)
    T = np.asarray(T, dtype=float)
    log_l = mbg_nbd_loglik(params, x, t_x, T)
    log_num = (gammaln(r + x + 1) - gammaln(r) + r * np.log(alpha)
               + gammaln(a + b) - gammaln(b) - gammaln(a + b + x)
               + gammaln(b + x) + np.log(b + x) - np.log(a + b + x)
               - (r + x + 1) * np.log(alpha + T))
    return t * np.exp(log_num - log_l)


# ---------------------------------------------------------------- Pareto/NBD

def _laguerre(n: int = 20):
    z, w = np.polynomial.laguerre.laggauss(n)
    return z, w


def pareto_nbd_loglik(params, x, t_x, T, n: int = 20):
    """Лог-правдоподобие Pareto/NBD через квадратуру Гаусса–Лагерра.

    λ ~ Gamma(r, rate α), μ ~ Gamma(s, rate β). Для клиента:
    L = ∫∫ λ^x [ μ/(λ+μ) e^{-(λ+μ)t_x} + λ/(λ+μ) e^{-(λ+μ)T} ] f(λ)f(μ).
    Двойной интеграл берётся по 20×20 узлам; это быстрее MCMC и достаточно
    точно для сравнения моделей по AIC.
    """
    r, alpha, s, beta = params
    x = np.asarray(x, dtype=float)
    t_x = np.asarray(t_x, dtype=float)
    T = np.asarray(T, dtype=float)
    z, w = _laguerre(n)

    lam = (z / alpha)[:, None]
    mu = (z / beta)[None, :]
    logw = ((np.log(w) - gammaln(r) + (r - 1) * np.log(z))[:, None]
            + (np.log(w) - gammaln(s) + (s - 1) * np.log(z))[None, :])
    log_sum = np.log(lam + mu)
    rate_sum = lam + mu

    log_lam = np.log(lam)[None, :, :]
    log_mu = np.log(mu)[None, :, :]
    base = x[:, None, None] * log_lam - log_sum[None, :, :]
    term1 = log_mu - rate_sum[None, :, :] * t_x[:, None, None]
    term2 = log_lam - rate_sum[None, :, :] * T[:, None, None]
    inner = base + np.logaddexp(term1, term2)
    return logsumexp(logw[None, :, :] + inner, axis=(1, 2))


def pareto_nbd_expected_purchases(params, x, t_x, T, t, n: int = 20):
    """E[покупок в (T, T+t]] для Pareto/NBD; интегрировано по «жив/мёртв»."""
    r, alpha, s, beta = params
    x = np.asarray(x, dtype=float)
    t_x = np.asarray(t_x, dtype=float)
    T = np.asarray(T, dtype=float)
    z, w = _laguerre(n)

    lam = (z / alpha)[:, None]
    mu = (z / beta)[None, :]
    logw = ((np.log(w) - gammaln(r) + (r - 1) * np.log(z))[:, None]
            + (np.log(w) - gammaln(s) + (s - 1) * np.log(z))[None, :])
    log_sum = np.log(lam + mu)
    rate_sum = lam + mu

    log_lam = np.log(lam)[None, :, :]
    log_num_inner = ((x[:, None, None] + 2) * log_lam - log_sum[None, :, :]
                     - rate_sum[None, :, :] * T[:, None, None])
    log_num = logsumexp(logw[None, :, :] + log_num_inner, axis=(1, 2))
    log_den = pareto_nbd_loglik(params, x, t_x, T, n=n)
    return t * np.exp(log_num - log_den)


# ---------------------------------------------------------------- Gamma-Gamma

def gamma_gamma_loglik(params, x, m):
    """Лог-правдоподобие Gamma-Gamma для средних повторных чеков.

    m_i | ν_i ~ Gamma(p x_i, rate = ν_i x_i), ν_i ~ Gamma(q, rate γ).
    x_i — число повторных транзакций (у нас — повторных дней с покупкой).
    """
    p, q, gamma = params
    x = np.asarray(x, dtype=float)
    m = np.asarray(m, dtype=float)
    k = p * x
    return np.sum(k * np.log(x) + (k - 1) * np.log(m) + q * np.log(gamma)
                  + gammaln(k + q) - gammaln(k) - gammaln(q)
                  - (k + q) * np.log(gamma + x * m))


def gamma_gamma_mean(params):
    """Популяционное среднее чека pγ/(q−1); требует q > 1."""
    p, q, gamma = params
    assert q > 1, "q must exceed 1 for a finite Gamma-Gamma mean"
    return p * gamma / (q - 1)


def gamma_gamma_conditional_spend(params, x, m):
    """E[M | x, m] для x ≥ 1; для x = 0 — популяционное среднее.

    Ноль здесь означает «нет повторных покупок», а не «нулевой чек»:
    таким клиентам возвращается условное среднее, а не 0 (§5а-3).
    """
    p, q, gamma = params
    x = np.asarray(x, dtype=float)
    m = np.asarray(m, dtype=float)
    out = np.full_like(x, gamma_gamma_mean(params), dtype=float)
    mask = x > 0
    denom = q + p * x[mask] - 1
    assert (denom > 0).all(), "Gamma-Gamma denominator must be positive"
    out[mask] = p * (gamma + x[mask] * m[mask]) / denom
    return out


# ---------------------------------------------------------------- MLE

def _fit_mle(loglik, starts, bounds):
    """Минимизация −loglik по логарифмам параметров (все параметры > 0)."""
    def neg(logp):
        return -float(np.sum(loglik(np.exp(logp))))

    best = None
    for x0 in starts:
        res = minimize(
            neg, np.log(np.asarray(x0, dtype=float)), method="L-BFGS-B",
            bounds=[(np.log(lo), np.log(hi)) for lo, hi in bounds],
            options={"maxiter": 4000, "ftol": 1e-12},
        )
        if best is None or res.fun < best.fun:
            best = res
    if not best.success:
        raise RuntimeError(f"MLE did not converge: {best.message}")
    return np.exp(best.x), best


def fit_bgnbd(x, t_x, T, starts=None, bounds=None):
    x = np.asarray(x, dtype=float)
    t_x = np.asarray(t_x, dtype=float)
    T = np.asarray(T, dtype=float)
    starts = starts or [(0.5, 20.0, 0.5, 2.0), (1.0, 10.0, 0.1, 1.0),
                        (0.2, 50.0, 1.0, 3.0)]
    bounds = bounds or [(0.01, 20.0), (0.01, 1000.0), (0.01, 20.0), (0.01, 100.0)]
    return _fit_mle(lambda p: bgnbd_loglik(p, x, t_x, T), starts, bounds)


def fit_mbg_nbd(x, t_x, T, starts=None, bounds=None):
    x = np.asarray(x, dtype=float)
    t_x = np.asarray(t_x, dtype=float)
    T = np.asarray(T, dtype=float)
    starts = starts or [(0.5, 20.0, 0.5, 2.0), (1.0, 10.0, 0.1, 1.0),
                        (0.2, 50.0, 1.0, 3.0)]
    bounds = bounds or [(0.01, 20.0), (0.01, 1000.0), (0.01, 20.0), (0.01, 100.0)]
    return _fit_mle(lambda p: mbg_nbd_loglik(p, x, t_x, T), starts, bounds)


def fit_pareto_nbd(x, t_x, T, starts=None, bounds=None, n: int = 20):
    x = np.asarray(x, dtype=float)
    t_x = np.asarray(t_x, dtype=float)
    T = np.asarray(T, dtype=float)
    starts = starts or [(0.5, 20.0, 0.5, 10.0), (1.0, 10.0, 0.2, 5.0),
                        (0.2, 50.0, 1.0, 20.0)]
    bounds = bounds or [(0.01, 20.0), (0.01, 1000.0), (0.001, 20.0), (0.01, 100000.0)]
    return _fit_mle(lambda p: pareto_nbd_loglik(p, x, t_x, T, n=n), starts, bounds)


def fit_gamma_gamma(x, m, starts=None, bounds=None):
    x = np.asarray(x, dtype=float)
    m = np.asarray(m, dtype=float)
    starts = starts or [(1.0, 2.0, 100.0), (2.0, 5.0, 1000.0), (0.5, 1.5, 50.0)]
    bounds = bounds or [(0.01, 50.0), (1.01, 100.0), (0.01, 5000.0)]
    return _fit_mle(lambda p: gamma_gamma_loglik(p, x, m), starts, bounds)


def aic(loglik: float, n_params: int) -> float:
    return 2 * n_params - 2 * float(loglik)


# ---------------------------------------------------------------- горизонт и дисконт

def monthly_expected_purchases(purchase_fn, params, x, t_x, T,
                               horizon_months: int = 12):
    """Прирост ожидаемых покупок по 30-дневным месяцам, shape (H, n)."""
    cumulative = np.vstack([
        purchase_fn(params, x, t_x, T, DAYS_PER_MONTH * (k + 1))
        for k in range(horizon_months)
    ])
    return np.diff(cumulative, axis=0, prepend=np.zeros((1, cumulative.shape[1])))


def discounted_revenue(monthly_revenue, annual_rate: float, months_per_year: int = 12):
    """PV потока по месяцам; покупки внутри месяца — в середине месяца."""
    d_m = (1 + annual_rate) ** (1 / months_per_year) - 1
    factors = (1 + d_m) ** (-(np.arange(monthly_revenue.shape[0]) + 0.5))
    return (monthly_revenue * factors[:, None]).sum(axis=0)
