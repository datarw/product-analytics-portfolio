import numpy as np
import pandas as pd

from features import (
    build_target,
    extend_cumulative_revenue,
    extend_survival,
    pre_first_mask,
    split_transactions,
)


def make_tx():
    rows = [
        (1, "2010-01-01", False, 100.0),
        (1, "2010-02-01", False, 50.0),
        (1, "2010-12-05", False, 30.0),
        (2, "2009-12-01", True, -20.0),
        (2, "2010-03-01", False, 80.0),
        (2, "2011-01-01", False, 40.0),
    ]
    df = pd.DataFrame(rows, columns=["customer_id", "date", "is_cancel", "revenue"])
    df["date"] = pd.to_datetime(df["date"])
    return df


def test_pre_first_mask_keeps_rows_but_excludes_them():
    tx = make_tx()
    mask = pre_first_mask(tx)
    assert int(mask.sum()) == 1
    assert bool(mask.iloc[3])


def test_split_excludes_pre_first_from_both_windows():
    tx = make_tx()
    cal, hold = split_transactions(tx, cutoff=pd.Timestamp("2010-12-01"), horizon_months=12)
    assert len(cal) == 3
    assert len(hold) == 2
    assert (cal.revenue > 0).all()
    assert (hold.revenue > 0).all()


def test_build_target_frequency_from_raw_days():
    tx = make_tx()
    target = build_target(tx, pd.Index([1, 2]), cutoff=pd.Timestamp("2010-12-01"),
                          horizon_months=12)
    assert target.loc[1, "revenue_holdout"] == 30.0
    assert target.loc[2, "revenue_holdout"] == 40.0
    assert target.loc[1, "frequency_holdout"] == 0
    assert target.loc[2, "frequency_holdout"] == 0


def test_survival_curve_is_non_increasing():
    raw = pd.Series({0: 1.0, 1: 0.5, 2: 0.55})
    out = extend_survival(raw, max_age=4)
    assert (np.diff(out.to_numpy()) <= 1e-12).all()


def test_cumulative_revenue_is_non_decreasing():
    raw = pd.Series({0: 100.0, 1: 50.0, 2: 60.0})
    out = extend_cumulative_revenue(raw, max_age=4)
    assert (np.diff(out.to_numpy()) >= -1e-9).all()
