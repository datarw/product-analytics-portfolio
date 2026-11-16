from experiment import holm_adjust, srm_test, two_proportion_ztest, wilson_ci


def test_srm_balanced():
    res = srm_test([100, 100])
    assert res["p_value"] > 0.99


def test_srm_imbalanced():
    res = srm_test([120, 80])
    assert res["p_value"] < 0.05


def test_two_proportion_equal():
    res = two_proportion_ztest(100, 1000, 100, 1000)
    assert abs(res["diff"]) < 1e-12
    assert res["p_value"] > 0.99


def test_two_proportion_large_diff():
    res = two_proportion_ztest(100, 1000, 200, 1000)
    assert res["diff"] > 0
    assert res["p_value"] < 0.001


def test_wilson_ci_contains_estimate():
    lo, hi = wilson_ci(50, 100)
    assert lo < 0.5 < hi


def test_holm_adjust_is_monotone_and_bounded():
    adjusted = holm_adjust([0.01, 0.04, 0.03])
    assert (adjusted >= 0).all() and (adjusted <= 1).all()
    assert adjusted.min() >= 0.01
