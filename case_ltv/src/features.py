"""RFM-признаки и когортные таблицы.

Все признаки строятся как f(транзакции, где date < cutoff). cutoff передаётся
явно; по умолчанию — конец данных плюс одна секунда (вся история).

Конвенция lifetimes (журнал, 2026-09-13):
    frequency   число повторных покупок (дней с покупкой минус 1)
    recency     дней между первой и последней покупкой
    T           дней от первой покупки до cutoff
    monetary    средняя сумма по повторным покупкам (NaN при frequency = 0)
RFM-recency (дней с последней покупки до cutoff) называется recency_rfm.
Покупка = день, в который у клиента есть счёт без префикса C (как в lifetimes).
"""

import numpy as np
import pandas as pd

DAYS_IN_AGE_MONTH = 30

# --- split калибровка / holdout (журнал, 2026-11-14) ---
CUTOFF_DATE = pd.Timestamp("2010-12-01")
HORIZON_MONTHS = 12
B3_T_DAYS = 60          # окно раннего множителя B3
B3_FIT_MONTHS = 6       # множитель оценивается на 6 мес, переносится кривой C(k)


def _sales(tx: pd.DataFrame, cutoff: pd.Timestamp | None) -> pd.DataFrame:
    if cutoff is not None:
        tx = tx[tx["date"] < cutoff]
    return tx


def build_features(tx: pd.DataFrame, cutoff: pd.Timestamp | None = None) -> pd.DataFrame:
    """Таблица на клиента: frequency, recency, T, monetary, recency_rfm, net_revenue."""
    tx = _sales(tx, cutoff)
    end = cutoff if cutoff is not None else tx["date"].max() + pd.Timedelta(seconds=1)

    sales = tx[~tx["is_cancel"]]
    daily = (sales.assign(day=sales["date"].dt.normalize())
             .groupby(["customer_id", "day"])
             .agg(amount=("revenue", "sum"), first_ts=("date", "min"))
             .reset_index())

    g = daily.groupby("customer_id")
    first_ts = g["first_ts"].min()
    first_day, last_day = g["day"].min(), g["day"].max()
    n_days = g["day"].size()

    out = pd.DataFrame({
        "first_purchase": first_ts,
        "frequency": n_days - 1,
        "recency": (last_day - first_day).dt.days,
        "T": (end - first_day).dt.days,
        "recency_rfm": (end - last_day).dt.days,
    })

    repeat = daily[daily["day"] > daily["customer_id"].map(first_day)]
    out["monetary"] = repeat.groupby("customer_id")["amount"].mean().reindex(out.index)

    out["sales_revenue"] = g["amount"].sum()
    out["net_revenue"] = tx.groupby("customer_id")["revenue"].sum().reindex(out.index)
    return out


def age_months(tx: pd.DataFrame, first_ts: pd.Series) -> pd.Series:
    """Возраст покупки в 30-дневных блоках от первой покупки клиента."""
    delta = tx["date"] - tx["customer_id"].map(first_ts)
    return (delta.dt.days // DAYS_IN_AGE_MONTH).astype("int64")


def cohort_tables(tx: pd.DataFrame, feats: pd.DataFrame, max_age: int = 12,
                  include_cancellations: bool = False):
    """Retention и revenue по возрасту, когорты — месяц первой покупки.

    Окно возраста k клиента наблюдается полностью, если first + 30(k+1) дней
    не позже конца данных. Ячейка когорты заполняется, только если полностью
    наблюдается каждый её член; иначе NaN (цензурирование, не ноль).

    include_cancellations=True считает revenue нетто (с отменами), но active
    по-прежнему считается по продажам: отмена не делает клиента активным.
    """
    end = tx["date"].max()
    sales = tx[~tx["is_cancel"]].copy()
    first = feats["first_purchase"]
    sales["age"] = age_months(sales, first)
    sales["cohort"] = sales["customer_id"].map(first).dt.to_period("M")
    rev_source = (tx if include_cancellations else sales).copy()
    rev_source["age"] = age_months(rev_source, first)
    rev_source["cohort"] = rev_source["customer_id"].map(first).dt.to_period("M")

    size = feats.groupby(feats["first_purchase"].dt.to_period("M")).size().rename("size")
    last_first = first.groupby(first.dt.to_period("M")).max()

    active = (sales[sales["age"] <= max_age].groupby(["cohort", "age"])["customer_id"]
              .nunique().unstack(fill_value=0).reindex(columns=range(max_age + 1), fill_value=0))
    rev = (rev_source[rev_source["age"] <= max_age].groupby(["cohort", "age"])["revenue"]
           .sum().unstack(fill_value=0.0).reindex(columns=range(max_age + 1), fill_value=0.0))

    complete = pd.DataFrame(
        {k: (last_first + pd.Timedelta(days=DAYS_IN_AGE_MONTH * (k + 1))) <= end
         for k in range(max_age + 1)})
    retention = active.div(size, axis=0).where(complete)
    revenue_per_customer = rev.div(size, axis=0).where(complete)
    return size, retention, revenue_per_customer


def calendar_retention(tx: pd.DataFrame, feats: pd.DataFrame, max_age: int = 12) -> pd.DataFrame:
    """Для сравнения: retention по календарным месяцам (как делают по привычке)."""
    sales = tx[~tx["is_cancel"]]
    first_m = feats["first_purchase"].dt.to_period("M")
    cohort = sales["customer_id"].map(first_m)
    k = (sales["date"].dt.year * 12 + sales["date"].dt.month) - (cohort.dt.year * 12 + cohort.dt.month)
    size = first_m.value_counts().sort_index()
    act = (sales.assign(cohort=cohort, k=k).query("k <= @max_age")
           .groupby(["cohort", "k"])["customer_id"].nunique().unstack(fill_value=0))
    last_m = sales["date"].max().to_period("M")
    ret = act.div(size.reindex(act.index), axis=0)
    # календарный месяц k наблюдается полностью, если он раньше последнего (неполного) месяца
    for c in ret.index:
        for kk in ret.columns:
            if c + kk >= last_m:
                ret.loc[c, kk] = np.nan
    return ret


# ---------------------------------------------------------------- split и target

def first_purchase_dates(tx: pd.DataFrame) -> pd.Series:
    """Дата первой покупки (счёт без префикса C) по всей истории."""
    return tx.loc[~tx["is_cancel"]].groupby("customer_id")["date"].min()


def pre_first_mask(tx: pd.DataFrame) -> pd.Series:
    """Строки раньше первой покупки клиента: 557 отмен из лево-цензурированного окна.

    Строки не удаляются из transactions.parquet; они исключаются только из
    lifecycle-расчётов (monetary, target), одинаково в калибровке и holdout.
    """
    first = first_purchase_dates(tx)
    return tx["date"] < tx["customer_id"].map(first)


def lifecycle_transactions(tx: pd.DataFrame) -> pd.DataFrame:
    """Транзакции внутри наблюдаемой жизни клиента (без строк до первой покупки)."""
    return tx.loc[~pre_first_mask(tx)].copy()


def holdout_end(cutoff: pd.Timestamp = CUTOFF_DATE,
                horizon_months: int = HORIZON_MONTHS) -> pd.Timestamp:
    return cutoff + pd.DateOffset(months=horizon_months)


def split_transactions(tx: pd.DataFrame, cutoff: pd.Timestamp = CUTOFF_DATE,
                       horizon_months: int = HORIZON_MONTHS):
    """Возвращает (калибровка, holdout) по одной переменной cutoff_date.

    Калибровка: date < cutoff. Holdout: cutoff <= date < cutoff + horizon.
    Строки до первой покупки клиента исключены из обоих окон.
    """
    end = holdout_end(cutoff, horizon_months)
    lc = lifecycle_transactions(tx)
    cal = lc[lc["date"] < cutoff].copy()
    hold = lc[(lc["date"] >= cutoff) & (lc["date"] < end)].copy()
    return cal, hold


def calibration_features(tx: pd.DataFrame, cutoff: pd.Timestamp = CUTOFF_DATE) -> pd.DataFrame:
    """RFM на калибровке: все агрегаты строго по date < cutoff."""
    return build_features(lifecycle_transactions(tx), cutoff=cutoff)


def calibration_revenue(tx: pd.DataFrame, cutoff: pd.Timestamp = CUTOFF_DATE) -> pd.Series:
    """Net-выручка калибровки (с отменами, без строк до первой покупки)."""
    cal = lifecycle_transactions(tx)
    cal = cal[cal["date"] < cutoff]
    return cal.groupby("customer_id")["revenue"].sum()


def build_target(tx: pd.DataFrame, customers, cutoff: pd.Timestamp = CUTOFF_DATE,
                 horizon_months: int = HORIZON_MONTHS) -> pd.DataFrame:
    """Таргет holdout: net-выручка, дни с покупкой и frequency.

    frequency_holdout = max(дни с покупкой - 1, 0), конвенция lifetimes.
    Считается из сырых транзакций, а не из производной колонки библиотеки.
    """
    end = holdout_end(cutoff, horizon_months)
    hold = lifecycle_transactions(tx)
    hold = hold[(hold["date"] >= cutoff) & (hold["date"] < end)]
    revenue = hold.groupby("customer_id")["revenue"].sum().reindex(customers, fill_value=0.0)
    sales = hold[~hold["is_cancel"]]
    days = (sales.assign(day=sales["date"].dt.normalize())
            .groupby("customer_id")["day"].nunique().reindex(customers, fill_value=0))
    frequency = (days - 1).clip(lower=0).astype("int64")
    return pd.DataFrame({
        "revenue_holdout": revenue.astype(float),
        "purchase_days_holdout": days.astype("int64"),
        "frequency_holdout": frequency,
    }, index=customers)


def customer_country(tx: pd.DataFrame) -> pd.Series:
    """Страна клиента — самая частая по строкам (для срезов калибровки)."""
    counts = (tx.groupby(["customer_id", "country"]).size()
              .rename("n").reset_index()
              .sort_values(["customer_id", "n"], ascending=[True, False]))
    return counts.drop_duplicates("customer_id").set_index("customer_id")["country"]


# ---------------------------------------------------------------- когортные кривые

def pooled_curve(per_customer: pd.DataFrame, size: pd.Series,
                 exclude_first: bool = True) -> pd.Series:
    """Взвешенная по размеру когорт средняя кривая; NaN там, где нет наблюдений."""
    use = per_customer.iloc[1:] if exclude_first else per_customer
    n = size.reindex(use.index)
    out = {}
    for k in use.columns:
        ok = use[k].notna() & n.notna() & (n > 0)
        out[k] = float((use[k][ok] * n[ok]).sum() / n[ok].sum()) if ok.any() else np.nan
    return pd.Series(out, dtype=float)


def monotone_decreasing(s: pd.Series) -> pd.Series:
    """Кумулятивный минимум: кривая выживания не может расти."""
    v = s.ffill().bfill().to_numpy(dtype=float)
    return pd.Series(np.minimum.accumulate(v), index=s.index)


def extend_survival(s: pd.Series, max_age: int = 24) -> pd.Series:
    """Продлевает убывающую кривую геометрическим хвостом до max_age."""
    s = monotone_decreasing(s)
    last = int(s.dropna().index.max())
    ratios = [s.loc[k] / s.loc[k - 1] for k in range(max(1, last - 2), last + 1)
              if s.loc[k - 1] > 0]
    rho = float(np.clip(np.median(ratios) if ratios else 1.0, 0.5, 1.0))
    out = s.reindex(range(max_age + 1)).astype(float)
    for k in range(last + 1, max_age + 1):
        out.loc[k] = out.loc[k - 1] * rho
    assert (np.diff(out.to_numpy()) <= 1e-12).all(), "survival curve must be non-increasing"
    return out


def extend_cumulative_revenue(incremental: pd.Series, max_age: int = 24) -> pd.Series:
    """Кумулятивная выручка на клиента; хвост — геометрическое затухание прироста.

    Прирост после последнего наблюдённого месяца не обнуляется: иначе кривая
    C(k) выходит на плато и B3 предсказывает ноль старым клиентам.
    """
    inc = incremental.reindex(range(13)).astype(float)
    last = int(inc.last_valid_index()) if inc.notna().any() else 0
    base = inc.loc[:last].fillna(0.0).clip(lower=0.0)
    ratios = [base.loc[k] / base.loc[k - 1] for k in range(max(1, last - 2), last + 1)
              if base.loc[k - 1] > 0]
    rho = float(np.clip(np.median(ratios) if ratios else 0.8, 0.5, 1.0))
    out = base.reindex(range(max_age + 1)).copy()
    for k in range(last + 1, max_age + 1):
        out.loc[k] = out.loc[k - 1] * rho
    cum = out.cumsum()
    assert (np.diff(cum.to_numpy()) >= -1e-9).all(), "cumulative revenue must be non-decreasing"
    return cum
