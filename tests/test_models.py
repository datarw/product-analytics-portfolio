import numpy as np

from models import (
    bgnbd_expected_purchases,
    bgnbd_loglik,
    gamma_gamma_conditional_spend,
    gamma_gamma_mean,
    mbg_nbd_loglik,
    pareto_nbd_loglik,
)


def test_bgnbd_zero_likelihood_closed_form():
    r, alpha = 1.0, 10.0
    params = (r, alpha, 0.5, 2.0)
    x = np.array([0.0])
    t_x = np.array([0.0])
    T = np.array([5.0])
    expected = r * (np.log(alpha) - np.log(alpha + T[0]))
    assert np.isclose(bgnbd_loglik(params, x, t_x, T)[0], expected)


def test_bgnbd_zero_expected_purchases_closed_form():
    r, alpha, t = 1.0, 10.0, 365.0
    params = (r, alpha, 0.5, 2.0)
    x = np.array([0.0])
    t_x = np.array([0.0])
    T = np.array([5.0])
    expected = t * r / (alpha + T[0])
    got = bgnbd_expected_purchases(params, x, t_x, T, t)[0]
    assert np.isclose(got, expected)


def test_gamma_gamma_mean_and_zero_client():
    params = (2.0, 3.0, 100.0)
    assert np.isclose(gamma_gamma_mean(params), 100.0)
    spend = gamma_gamma_conditional_spend(
        params, np.array([0.0, 1.0]), np.array([np.nan, 50.0])
    )
    assert np.isfinite(spend).all()
    assert (spend > 0).all()
    assert np.isclose(spend[0], 100.0)
    assert np.isclose(spend[1], 75.0)


def test_mbg_and_pareto_loglik_are_finite():
    params = (1.0, 10.0, 0.5, 2.0)
    x = np.array([1.0, 2.0])
    t_x = np.array([1.0, 2.0])
    T = np.array([3.0, 4.0])
    assert np.isfinite(mbg_nbd_loglik(params, x, t_x, T)).all()
    assert np.isfinite(pareto_nbd_loglik((1.0, 10.0, 0.5, 10.0), x, t_x, T, n=8)).all()
