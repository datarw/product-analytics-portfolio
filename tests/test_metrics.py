import numpy as np
import pandas as pd

from metrics import (
    calibration_ratio,
    gini,
    lorenz_points,
    shrink_predictions,
    smallest_segment_for_coverage,
    top_capture,
)


def test_gini_orders_scores():
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert gini(y, y) > 0.2
    assert gini(y, -y) < 0


def test_top_capture_perfect_and_random():
    y = np.array([10.0, 0.0, 0.0, 0.0])
    assert top_capture(y, y, 0.25) == 1.0
    assert 0.0 <= top_capture(y, -y, 0.25) <= 1.0


def test_calibration_ratio():
    assert calibration_ratio(np.array([1.0, 1.0]), np.array([2.0, 2.0])) == 2.0


def test_lorenz_endpoints():
    y = np.array([1.0, 2.0, 3.0])
    x, cum = lorenz_points(y, y)
    assert x[0] == 0 and x[-1] == 1
    assert cum[0] == 0 and cum[-1] == 1


def test_smallest_segment_for_coverage():
    y = np.array([10.0, 0.0, 0.0, 0.0])
    score = np.array([4.0, 3.0, 2.0, 1.0])
    res = smallest_segment_for_coverage(y, score, target=0.6)
    assert res["k"] == 1
    assert res["coverage"] == 1.0


def test_shrink_predictions_moves_to_group_mean():
    pred = pd.Series([10.0, 20.0], index=[1, 2])
    groups = pd.Series(["a", "a"], index=[1, 2])
    n_obs = pd.Series([1, 100], index=[1, 2])
    out = shrink_predictions(pred, groups, n_obs, k=10)
    assert out.loc[1] > pred.loc[1]
    assert out.loc[2] < pred.loc[2]
