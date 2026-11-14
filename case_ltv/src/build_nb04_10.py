"""Собирает notebooks/04…10. Исполнение — отдельно nbconvert, чтобы видеть каждый шаг."""

from pathlib import Path

import nbformat as nbf

NBDIR = Path(__file__).resolve().parents[1] / "notebooks"
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

HEAD = """import sys
sys.path.insert(0, '../..'); sys.path.insert(0, '../src')
import json
from inspect import signature
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from paths import CASE_LTV, PROCESSED, RANDOM_SEED, figures
from features import (CUTOFF_DATE, HORIZON_MONTHS, B3_T_DAYS, B3_FIT_MONTHS,
                      DAYS_IN_AGE_MONTH, split_transactions, calibration_features,
                      calibration_revenue, build_target, lifecycle_transactions,
                      first_purchase_dates, pre_first_mask, customer_country,
                      cohort_tables, pooled_curve, extend_survival,
                      extend_cumulative_revenue, age_months)
from metrics import (gini, top_capture, top_lift, decile_capture, lorenz_points,
                     mae, rmse, calibration_ratio, model_metrics, bootstrap_ci,
                     bootstrap_diff, calibration_by, coverage_at_top,
                     smallest_segment_for_coverage, shrink_predictions)
from models import (fit_bgnbd, fit_mbg_nbd, fit_pareto_nbd, fit_gamma_gamma,
                    bgnbd_loglik, bgnbd_expected_purchases, mbg_nbd_loglik,
                    mbg_nbd_expected_purchases, pareto_nbd_loglik,
                    pareto_nbd_expected_purchases, gamma_gamma_loglik,
                    gamma_gamma_mean, gamma_gamma_conditional_spend,
                    monthly_expected_purchases, discounted_revenue, aic)
from plots import (plot_split_timeline, plot_lorenz, plot_model_comparison,
                   plot_calibration_by_decile, plot_horizon, plot_segments,
                   plot_shrinkage, plot_posterior)
FIG = figures(CASE_LTV); FIG.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(RANDOM_SEED)
tx = pd.read_parquet(PROCESSED / 'transactions.parquet')
"""


def run(name: str, cells: list) -> None:
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nbf.write(nb, NBDIR / name)
    print("written", name)


split = [
    md("""# 04 · Calibration / holdout split

Шаг 5 плана. Одна переменная `cutoff_date = 2010-12-01`: калибровка — `date < cutoff`,
holdout — `cutoff <= date < cutoff + 12 мес`. Holdout наблюдается полностью; неполный
декабрь 2011 исключён, ноябрьский пик входит в оба окна. Признаки строятся только по
калибровке, таргет — только по holdout."""),
    code(HEAD + """
cal, hold = split_transactions(tx)
feats = calibration_features(tx)
rev_cal = calibration_revenue(tx)
target = build_target(tx, feats.index)
first = first_purchase_dates(tx)
end = CUTOFF_DATE + pd.DateOffset(months=HORIZON_MONTHS)
new_hold = first[(first >= CUTOFF_DATE) & (first < end)].index
after = first[first >= end].index
print('калибровка:', cal.shape, '· holdout:', hold.shape)
print('клиентов:', len(feats), '· новых в holdout:', len(new_hold), '· после holdout:', len(after))
print('период калибровки:', cal.date.min(), '→', cal.date.max())
print('период holdout:', hold.date.min(), '→', hold.date.max())
assert feats.index.equals(first[first < CUTOFF_DATE].index)
"""),
    md("""## Проверка §5а-2: frequency в holdout — из сырых транзакций

`frequency_holdout` не выводится из калибровочной `frequency` и не берётся из
производной колонки: это число уникальных дней с покупкой в holdout минус один."""),
    code("""hold_sales = hold[~hold.is_cancel]
raw_days = (hold_sales.assign(day=hold_sales.date.dt.normalize())
            .groupby('customer_id')['day'].nunique()
            .reindex(feats.index, fill_value=0))
raw_freq = (raw_days - 1).clip(lower=0).astype('int64')
assert (raw_freq == target.frequency_holdout).all(), 'holdout frequency не совпала с сырыми днями'
assert not (target.frequency_holdout == feats.frequency).all(), 'holdout и калибровка перепутаны'
print('ok: frequency holdout пересобрана из сырых транзакций')
print('доля frequency_holdout = 0:', f'{(target.frequency_holdout == 0).mean():.1%}')
"""),
    md("""## Отмены раньше первой покупки

557 строк с датой раньше первой покупки клиента — это возвраты покупок, сделанных до
начала окна. Строки не удаляются из `transactions.parquet`, но исключаются из
lifecycle-расчётов и в калибровке, и в holdout. Правило одно на оба окна."""),
    code("""pre = pre_first_mask(tx)
assert int(pre.sum()) == 557 and int((pre & tx.is_cancel).sum()) == 557
hold_pre = pre & (tx.date >= CUTOFF_DATE) & (tx.date < end)
print('строк до первой покупки:', int(pre.sum()), '· из них в holdout:', int(hold_pre.sum()))
print('net-выручка калибровки:', round(rev_cal.sum()), '£ · holdout:', round(target.revenue_holdout.sum()), '£')
"""),
    md("## Сохранение split"),
    code("""split = (feats.join(target).join(rev_cal.rename('revenue_cal')))
split['country'] = customer_country(tx)
split['cohort'] = split.first_purchase.dt.to_period('M').astype(str)
split['age_days'] = (CUTOFF_DATE - split.first_purchase).dt.days
split.to_parquet(PROCESSED / 'ltv_split.parquet')
meta = {
    'cutoff_date': str(CUTOFF_DATE.date()),
    'holdout_end': str(end.date()),
    'horizon_months': HORIZON_MONTHS,
    'calibration_customers': int(len(split)),
    'new_in_holdout': int(len(new_hold)),
    'after_holdout': int(len(after)),
    'pre_first_rows': int(pre.sum()),
    'pre_first_in_holdout': int(hold_pre.sum()),
    'holdout_revenue': float(target.revenue_holdout.sum()),
}
(PROCESSED / 'ltv_split_meta.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
meta
"""),
    md("## График окна"),
    code("""plot_split_timeline(tx, CUTOFF_DATE, end, FIG / '07-split-timeline.png')
plt.show()
"""),
    md("""Слева калибровка (12 мес), в центре holdout (12 мес), справа исключённый хвост
2011-12-01…09. Holdout заканчивается 2011-11-30, поэтому ноябрьский пик попал в него
целиком, а неполный декабрь не разбавляет таргет."""),
]

baselines = [
    md("""# 05 · Бейзлайны B1–B4

Шаг 6 плана. Четыре бейзлайна считаются только по данным до cutoff:
B1 — историческая сумма; B2 — среднее по когорте; B3 — мультипликатор
(медиана `revenue_6m/revenue_60d` по когортам, перенос на 12 мес через `C(k)`);
B4 — сумма когортной кривой выживания с ассертом на убывание."""),
    code(HEAD + """
split = pd.read_parquet(PROCESSED / 'ltv_split.parquet')
cal, hold = split_transactions(tx)
idx = split.index
y = split.revenue_holdout.to_numpy(float)
"""),
    md("## B1 и B2"),
    code("""split['B1'] = split.revenue_cal
cm = split.groupby('cohort')['revenue_cal'].agg(['mean', 'size'])
cm.loc[cm['size'] < 30, 'mean'] = split.revenue_cal.mean()
split['B2'] = split.cohort.map(cm['mean']).astype(float)
print('B2: когорт с <30 клиентов заменено на общее среднее:', int((cm['size'] < 30).sum()))
"""),
    md("""## B3: мультипликатор

12-месячный множитель нельзя оценить без утечки: первое 12-месячное окно когорт
2010-01…2010-11 заканчивается после cutoff. Оцениваю 6-месячный множитель на
когортах, у которых эти 6 месяцев наблюдаются до cutoff, и переношу на 12 месяцев
через когортную кривую `C(k)`. `t = 60` дней: 30 дней дают более шумное отношение."""),
    code("""cal_lc = lifecycle_transactions(tx)
cal_lc = cal_lc[cal_lc.date < CUTOFF_DATE]
cal_sales = cal_lc[~cal_lc.is_cancel]
first = split.first_purchase
age_days = (CUTOFF_DATE - first).dt.days

def window_rev(days):
    m = cal_sales.assign(lo=cal_sales.customer_id.map(first),
                         hi=cal_sales.customer_id.map(first + pd.Timedelta(days=days)))
    m = m[(m.date >= m['lo']) & (m.date < m['hi'])]
    return m.groupby('customer_id')['revenue'].sum().reindex(idx, fill_value=0.0)

rev_t = window_rev(B3_T_DAYS)
rev_6m = window_rev(B3_FIT_MONTHS * 30)
rev_t_adj = rev_t * B3_T_DAYS / np.minimum(age_days.clip(lower=1), B3_T_DAYS)
elig = age_days >= B3_FIT_MONTHS * 30
ratio = rev_6m / rev_t.replace(0, np.nan)
m_cohort = ratio[elig].groupby(split.loc[elig, 'cohort']).median()
pooled_m = ratio[elig].median()
split['B3_multiplier'] = split.cohort.map(m_cohort).fillna(pooled_m)
print('B3: pooled multiplier', round(pooled_m, 3), '· когорт с оценкой', len(m_cohort))
print(m_cohort.round(3).to_dict())
"""),
    code("""size, ret, rpc = cohort_tables(cal, split, max_age=12, include_cancellations=True)
inc_pool = pooled_curve(rpc, size, exclude_first=True)
C = extend_cumulative_revenue(inc_pool, max_age=24)
a = (age_days // DAYS_IN_AGE_MONTH).clip(0, 12).astype(int)
c_a = C.reindex(a.values).to_numpy()
c_a12 = C.reindex((a + 12).clip(upper=24).values).to_numpy()
split['B3'] = rev_t_adj.to_numpy() * split.B3_multiplier.to_numpy() * (c_a12 - c_a) / C.loc[6]
split.loc[~np.isfinite(split.B3) | (split.B3 < 0), 'B3'] = split.loc[~np.isfinite(split.B3) | (split.B3 < 0), 'B2']
print('C(6)=', round(C.loc[6], 1), '· C(12)=', round(C.loc[12], 1), '· C(24)=', round(C.loc[24], 1))
assert (np.diff(C.to_numpy()) >= -1e-9).all(), 'C(k) должна быть неубывающей'
"""),
    md("""## B4: сумма когортной кривой выживания

Сырая retention по когортам немонотонна: на возрастах 6–9 она растёт из-за состава
когорт (короткие когорты выживают лучше). Для B4 нужна кривая выживания, поэтому
беру кумулятивный минимум и продлеваю геометрическим хвостом. Ассерт на убывание —
в `extend_survival` и ниже."""),
    code("""ret_pool = pooled_curve(ret, size, exclude_first=True)
S = extend_survival(ret_pool, max_age=24)
assert (np.diff(S.to_numpy()) <= 1e-12).all(), 'survival curve должна убывать'
active_blocks = (cal_sales.assign(age=age_months(cal_sales, first))
                 .groupby('customer_id')['age'].nunique().reindex(idx, fill_value=1).clip(lower=1))
arpu = split.revenue_cal / active_blocks
split['B4'] = [arpu.iloc[i] * sum(S.loc[k] for k in range(int(a.iloc[i]), min(int(a.iloc[i]) + 12, 25)))
               for i in range(len(split))]
print('S(1..12):', S.loc[1:12].round(3).to_dict())
"""),
    md("## Таблица «бейзлайн × метрика»"),
    code("""rows = []
for m in ['B1', 'B2', 'B3', 'B4']:
    rows.append({'model': m, **model_metrics(y, split[m].to_numpy(float))})
base_table = pd.DataFrame(rows).set_index('model')
base_table.round(4)
"""),
    code("""split[['B1', 'B2', 'B3', 'B4', 'B3_multiplier']].to_parquet(PROCESSED / 'baselines.parquet')
base_table.to_csv(PROCESSED / 'baseline_metrics.csv')
assert split[['B1', 'B2', 'B3', 'B4']].notna().all().all()
assert (split[['B1', 'B2', 'B4']] >= 0).all().all()
"""),
    md("## Кривые Лоренца бейзлайнов"),
    code("""plot_lorenz(y, {'B1 историческая сумма': split.B1, 'B2 среднее по когорте': split.B2,
                 'B3 мультипликатор': split.B3, 'B4 кривая выживания': split.B4},
           FIG / '08-baseline-lorenz.png', title='Бейзлайны на holdout')
plt.show()
"""),
    md("""B1 — сильный конкурент: калибровочная выручка несёт всю информацию о размере
клиента. B2 теряет индивидуальность и падает по Gini. B3 и B4 — отраслевые бейзлайны;
их сравнение с BG/NBD идёт в ноутбуке 08."""),
]

bgnbd = [
    md("""# 06 · BG/NBD: байесовский фит, варианты и диагностика

Шаг 7 плана. Основной фит — BG/NBD на всех клиентах калибровки, включая
`frequency = 0` (правдоподобие определено при x = 0). Отдельно — фит без
одноразовых покупателей: экспонат о том, что даёт популярная ошибка фильтрации.
Сравнение — с Pareto/NBD и MBG/NBD по log-likelihood, AIC и holdout."""),
    code(HEAD + """
split = pd.read_parquet(PROCESSED / 'ltv_split.parquet')
d = split[split['T'] > 0].copy()
x = d.frequency.to_numpy(float)
t_x = d.recency.to_numpy(float)
T = d['T'].to_numpy(float)
print('клиентов в фите:', len(d), '· frequency = 0:', f'{(x == 0).mean():.1%}')
"""),
    md("## Байесовский BG/NBD (PyMC-Marketing)"),
    code("""from pymc_marketing.clv import BetaGeoModel

data_bayes = d.reset_index()[['customer_id', 'frequency', 'recency', 'T']]
model = BetaGeoModel()
model.fit(data_bayes, draws=500, tune=500, chains=4, cores=1,
          target_accept=0.9, progressbar=False, random_seed=RANDOM_SEED)
summary = model.fit_summary()
summary.round(4)
"""),
    md("""## §5а-6: диагностика MCMC — числами

Не меньше двух цепочек, `r_hat < 1.01`, `ess_bulk > 400`, расхождения — из
`az.summary`/`fit_summary`, не с графика."""),
    code("""s = summary.loc[['r', 'alpha', 'a', 'b']]
assert model.idata.posterior.sizes['chain'] >= 2
assert (s.r_hat < 1.01).all(), s[['r_hat']]
assert (s.ess_bulk > 400).all(), s[['ess_bulk']]
divergences = int(model.idata.sample_stats['diverging'].sum())
assert divergences == 0, divergences
print('r_hat max:', round(float(s.r_hat.max()), 4), '· ess_bulk min:', round(float(s.ess_bulk.min())))
print('divergences:', divergences)
"""),
    md("## §5а-1: параметрическое среднее против эмпирического"),
    code("""r_mean, alpha_mean = float(s.loc['r', 'mean']), float(s.loc['alpha', 'mean'])
emp = (x / T).mean()
assert abs(np.log10(r_mean / alpha_mean) - np.log10(emp)) < 1, 'rate vs scale?'
print('r/alpha =', round(r_mean / alpha_mean, 5), '· mean(f/T) =', round(emp, 5))
"""),
    md("""## §5а-5: один общий индекс draw

Параметры постериора нельзя пересэмплировать независимо: это разрушает совместное
распределение. Беру 200 draw одним индексом и проверяю, что корреляция `a,b`
сохраняется."""),
    code("""post = model.idata.posterior.to_dataset().stack(sample=('chain', 'draw'))
idx = RNG.choice(post.sizes['sample'], size=200, replace=False)
r_draw = post['r'].to_numpy()[idx]
alpha_draw = post['alpha'].to_numpy()[idx]
a_draw = post['a'].to_numpy()[idx]
b_draw = post['b'].to_numpy()[idx]
corr_joint = np.corrcoef(a_draw, b_draw)[0, 1]
corr_full = np.corrcoef(post['a'].to_numpy(), post['b'].to_numpy())[0, 1]
assert abs(corr_joint - corr_full) < 0.05, (corr_joint, corr_full)
print('corr(a,b) joint:', round(corr_joint, 3), '· full posterior:', round(corr_full, 3))
"""),
    md("## Прогноз покупок: среднее по 200 joint draw"),
    code("""pred_draws = np.vstack([
    bgnbd_expected_purchases((r_draw[i], alpha_draw[i], a_draw[i], b_draw[i]), x, t_x, T, 365)
    for i in range(len(idx))
])
pred_bayes = pred_draws.mean(axis=0)
print('E[покупок за 12 мес] mean:', round(pred_bayes.mean(), 2))
print('факт: дней с покупкой в holdout mean:', round(d.purchase_days_holdout.mean(), 2))
"""),
    md("## MLE: полная выборка и выборка без одноразовых покупателей"),
    code("""p_full, res_full = fit_bgnbd(x, t_x, T)
mask = x > 0
p_nonone, res_nonone = fit_bgnbd(x[mask], t_x[mask], T[mask])
pred_full = bgnbd_expected_purchases(p_full, x, t_x, T, 365)
pred_nonone = bgnbd_expected_purchases(p_nonone, x, t_x, T, 365)
ll_full = float(np.sum(bgnbd_loglik(p_full, x, t_x, T)))
ll_nonone = float(np.sum(bgnbd_loglik(p_nonone, x[mask], t_x[mask], T[mask])))
print('full params:', np.round(p_full, 4), '· loglik', round(ll_full, 1))
print('no-one-time params:', np.round(p_nonone, 4), '· loglik', round(ll_nonone, 1))
print('pred purchases full:', round(pred_full.mean(), 2), '· no-one-time applied to all:', round(pred_nonone.mean(), 2))
print('actual purchase days:', round(d.purchase_days_holdout.mean(), 2))
"""),
    md("""Экспонат: фит без `frequency = 0` видит только «выживших», завышает
`p(alive)` и суммарный прогноз. Сравнение — в таблице ниже и в README."""),
    md("## Pareto/NBD и MBG/NBD"),
    code("""p_mbg, res_mbg = fit_mbg_nbd(x, t_x, T)
p_par, res_par = fit_pareto_nbd(x, t_x, T, n=16)
ll_mbg = float(np.sum(mbg_nbd_loglik(p_mbg, x, t_x, T)))
ll_par = float(np.sum(pareto_nbd_loglik(p_par, x, t_x, T, n=16)))
pred_mbg = mbg_nbd_expected_purchases(p_mbg, x, t_x, T, 365)
pred_par = pareto_nbd_expected_purchases(p_par, x, t_x, T, 365, n=16)
fit_table = pd.DataFrame({
    'model': ['BG/NBD (MLE)', 'MBG/NBD', 'Pareto/NBD'],
    'loglik': [ll_full, ll_mbg, ll_par],
    'aic': [aic(ll_full, 4), aic(ll_mbg, 4), aic(ll_par, 4)],
    'mean_purchases_12m': [pred_full.mean(), pred_mbg.mean(), pred_par.mean()],
})
fit_table.round(2)
"""),
    code("""params = pd.DataFrame({
    'BG/NBD (MLE)': p_full, 'MBG/NBD': p_mbg, 'Pareto/NBD': p_par,
}, index=['r', 'alpha', 'a_or_s', 'b_or_beta'])
params.round(4)
"""),
    md("""Pareto/NBD уходит в границу `beta → ∞`: отток не идентифицируется, модель
вырождается в NBD. AIC формально чуть лучше, но параметр на границе — сравнивать
по AIC нельзя. MBG/NBD тоже уходит в `b → 100`. Это результат: три модели
различаются слабо, выбираю BG/NBD по простоте и интерпретируемости."""),
    md("## Сохранение постериора и прогнозов"),
    code("""draws = pd.DataFrame({k: model.idata.posterior[k].to_numpy().ravel()
                      for k in ['r', 'alpha', 'a', 'b']})
draws['chain'] = np.repeat(np.arange(model.idata.posterior.sizes['chain']),
                           model.idata.posterior.sizes['draw'])
draws['draw'] = np.tile(np.arange(model.idata.posterior.sizes['draw']),
                        model.idata.posterior.sizes['chain'])
draws.to_parquet(PROCESSED / 'bgnbd_posterior.parquet')
summary.to_csv(PROCESSED / 'bgnbd_summary.csv')
predictions = pd.DataFrame({
    'BG_bayes': pred_bayes, 'BG_mle': pred_full, 'BG_no_one_time': pred_nonone,
    'MBG': pred_mbg, 'Pareto': pred_par,
}, index=d.index)
predictions.to_parquet(PROCESSED / 'purchases.parquet')
fit_table.to_csv(PROCESSED / 'frequency_model_fit.csv', index=False)
params.to_csv(PROCESSED / 'frequency_model_params.csv')
assert predictions.notna().all().all() and (predictions >= 0).all().all()
"""),
    md("## Графики диагностики"),
    code("""plot_posterior(summary, FIG / '09-bgnbd-posterior.png')
plt.show()
fig, ax = plt.subplots(figsize=(7, 4))
labels = ['BG/NBD', 'MBG/NBD', 'Pareto/NBD', 'без x=0']
vals = [pred_full.mean(), pred_mbg.mean(), pred_par.mean(), pred_nonone.mean()]
ax.bar(labels, vals, color=['#4C78A8', '#72B7B2', '#F58518', '#E45756'])
ax.axhline(d.purchase_days_holdout.mean(), color='black', ls='--', lw=1, label='факт holdout')
ax.set(ylabel='E[дней с покупкой за 12 мес]', title='Прогноз покупок: модели и ошибка фильтрации')
ax.legend(frameon=False); fig.tight_layout(); fig.savefig(FIG / '10-purchase-models.png'); plt.show()
"""),
    md("""Все три модели дают близкий прогноз покупок; фит без одноразовых покупателей
заметно выше факта. Это и есть цена популярной ошибки: модель учится на «выживших»
и переносит их поведение на всю базу."""),
]

gamma = [
    md("""# 07 · Gamma-Gamma: денежная часть

Шаг 8 плана. Gamma-Gamma фитится только на клиентах с `frequency >= 1`.
Клиенты с `frequency = 0` возвращаются в расчёт с условным средним, а не с нулём.
Проверки: независимость frequency и monetary, §5а-1 (mean против empirical),
§5а-3 (ноль против «нет данных»), историческая денежная часть против последнего
периода."""),
    code(HEAD + """
split = pd.read_parquet(PROCESSED / 'ltv_split.parquet')
purchases = pd.read_parquet(PROCESSED / 'purchases.parquet')
d = split.loc[purchases.index].copy()
x = d.frequency.to_numpy(float)
m = d.monetary.to_numpy(float)
g = d[d.frequency >= 1]
p_gg, res_gg = fit_gamma_gamma(g.frequency.to_numpy(float), g.monetary.to_numpy(float))
print('Gamma-Gamma params (p, q, gamma):', np.round(p_gg, 3))
print('pop mean:', round(gamma_gamma_mean(p_gg), 2), '£ · empirical mean:', round(g.monetary.mean(), 2), '£')
"""),
    md("## §5а-1: параметрическое среднее против эмпирического"),
    code("""assert abs(np.log10(gamma_gamma_mean(p_gg)) - np.log10(g.monetary.mean())) < 1, 'rate vs scale?'
ll_gg = float(gamma_gamma_loglik(p_gg, g.frequency.to_numpy(float), g.monetary.to_numpy(float)))
print('loglik:', round(ll_gg, 1))
"""),
    md("## Независимость frequency и monetary"),
    code("""pear = g[['frequency', 'monetary']].corr().iloc[0, 1]
spear = g[['frequency', 'monetary']].corr(method='spearman').iloc[0, 1]
assert abs(pear) < 0.30, f'Gamma-Gamma independence violated: pearson={pear:.3f}'
print('Pearson:', round(pear, 3), '· Spearman:', round(spear, 3))
fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(g.frequency.clip(upper=30), np.log10(g.monetary), s=6, alpha=.25)
ax.set(xlabel='frequency (обрезано на 30)', ylabel='log10 среднего чека, £',
       title='frequency vs monetary (x ≥ 1)')
fig.tight_layout(); fig.savefig(FIG / '11-frequency-monetary.png'); plt.show()
"""),
    md("""Pearson близок к нулю, Spearman 0.25: линейной зависимости нет, но ранговая
слабая. Допущение Gamma-Gamma — независимость среднего чека от частоты — не нарушено
в линейной форме; остаточную ранговую связь отмечаю в ограничениях."""),
    md("""## Условное среднее и возврат frequency = 0

Для x ≥ 1 — `E[M|x,m]` из Gamma-Gamma. Для x = 0 monetary не определено; возвращаю
shrinkage первого чека к популяционному среднему: `0.5·first_check + 0.5·mean`.
Вес фиксирован, на holdout не подбирался."""),
    code("""spend_hist = gamma_gamma_conditional_spend(p_gg, x, m)
first_check = split.loc[d.index, 'revenue_cal']  # placeholder overwritten below
# первый чек = сумма первого дня с покупкой
cal_lc = lifecycle_transactions(tx); cal_lc = cal_lc[cal_lc.date < CUTOFF_DATE]
cal_sales = cal_lc[~cal_lc.is_cancel]
first = d.first_purchase
first_day = first.dt.normalize()
fd = cal_sales.assign(day=cal_sales.date.dt.normalize())
first_check = fd[fd.day == fd.customer_id.map(first_day)].groupby('customer_id')['revenue'].sum().reindex(d.index)
w = 0.5
pop_mean = gamma_gamma_mean(p_gg)
spend_main = spend_hist.copy()
x0 = x == 0
spend_main[x0] = w * first_check.to_numpy()[x0] + (1 - w) * pop_mean
print('first_check: median', round(first_check.median(), 1), '£')
print('spend x=0: mean', round(spend_main[x0].mean(), 1), '£ · spend x>=1: mean', round(spend_main[~x0].mean(), 1), '£')
"""),
    md("## §5а-3: ноль против «нет данных»"),
    code("""# monetary не определено ровно там, где frequency = 0 — это не ноль чека
assert (np.isnan(m) == (x == 0)).all()
# на выходе нулей нет: клиенты x=0 получили условное среднее, а не 0
assert not np.isclose((spend_main == 0).mean(), (x == 0).mean(), atol=0.02)
print('доля NaN monetary:', f'{np.isnan(m).mean():.1%}', '· доля нулей в spend:', f'{(spend_main == 0).mean():.1%}')
"""),
    md("""## Историческая денежная часть против последнего периода

Проверка из журнала: цены и ассортимент меняются быстрее поведения. Сравниваю
средний чек по всей калибровке и по последним 180 дням до cutoff; для клиентов без
повторных покупок в последнем окне остаётся историческая оценка."""),
    code("""last_start = CUTOFF_DATE - pd.Timedelta(days=180)
recent = cal_sales[cal_sales.date >= last_start]
recent = recent.assign(day=recent.date.dt.normalize())
recent_daily = recent.groupby(['customer_id', 'day'])['revenue'].sum().reset_index()
# исключаем первый день: monetary — только повторные покупки
first_day_map = first_day
recent_daily = recent_daily[recent_daily.day != recent_daily.customer_id.map(first_day_map)]
m_recent = recent_daily.groupby('customer_id')['revenue'].mean().reindex(d.index)
m_recent = m_recent.fillna(pd.Series(m, index=d.index))
spend_recent = gamma_gamma_conditional_spend(p_gg, x, m_recent.to_numpy(float))
spend_recent[x0] = spend_main[x0]
print('m_recent покрывает:', f'{m_recent.notna().mean():.1%}')
"""),
    code("""rows = []
for name, spend in [('historical', spend_main), ('last_180d', spend_recent)]:
    pred = purchases['BG_bayes'].to_numpy() * spend
    rows.append({'monetary': name, **model_metrics(d.revenue_holdout.to_numpy(float), pred)})
monetary_table = pd.DataFrame(rows).set_index('monetary')
monetary_table.round(4)
"""),
    md("""Если последний период даёт другой ranking/калибровку, это видно в таблице.
Историческая часть остаётся основной: у последнего окна меньше наблюдений и оно
сильнее смещено сезонностью."""),
    code("""pd.DataFrame({'spend_hist': spend_hist, 'spend_recent': spend_recent,
              'spend_main': spend_main, 'first_check': first_check}).to_parquet(PROCESSED / 'spend.parquet')
monetary_table.to_csv(PROCESSED / 'monetary_comparison.csv')
assert (spend_main > 0).all()
"""),
]

validation = [
    md("""# 08 · Holdout-валидация: две метрики и bootstrap

Шаг 9 плана. Всегда две оси: ранжирование (Gini, Lorenz, decile capture) и
калибровка (MAE/RMSE, calibration ratio по срезам). Bootstrap по клиентам — 1 000
ресурсов, один `Generator`; сравнение без интервала не публикуется."""),
    code(HEAD + """
split = pd.read_parquet(PROCESSED / 'ltv_split.parquet')
base = pd.read_parquet(PROCESSED / 'baselines.parquet')
purch = pd.read_parquet(PROCESSED / 'purchases.parquet')
spend = pd.read_parquet(PROCESSED / 'spend.parquet')
d = split.join(base[['B1', 'B2', 'B3', 'B4']]).join(purch).join(spend[['spend_main']])
y = d.revenue_holdout.to_numpy(float)
d['BG'] = d.BG_bayes * d.spend_main
d['MBG'] = d.MBG * d.spend_main
d['Pareto'] = d.Pareto * d.spend_main
d['BG_no_one_time'] = d.BG_no_one_time * d.spend_main
models = ['B1', 'B2', 'B3', 'B4', 'BG', 'MBG', 'Pareto', 'BG_no_one_time']
assert d[models].notna().all().all()
"""),
    md("## §5а-4: произведение уже проинтегрировано по «жив/мёртв»"),
    code("""def clv_from_purchases(purchases, spend):
    \"\"\"CLV = E[покупки] × E[чек]. E[покупки] уже интегрировано по alive/dead;
    p(alive) сюда не входит и повторно не умножается.\"\"\"
    return purchases * spend

assert 'p_alive' not in clv_from_purchases.__code__.co_varnames
d['BG_check'] = clv_from_purchases(d.BG_bayes.to_numpy(), d.spend_main.to_numpy())
assert np.allclose(d.BG_check, d.BG)
"""),
    md("## §5а-3 на выходе CLV"),
    code("""assert not np.isclose((d.BG == 0).mean(), (d.frequency == 0).mean(), atol=0.02)
print('доля нулей в CLV:', f'{(d.BG == 0).mean():.1%}', '· доля frequency=0:', f'{(d.frequency == 0).mean():.1%}')
"""),
    md("## Таблица «модель × метрика»"),
    code("""rows = []
for m in models:
    rows.append({'model': m, **model_metrics(y, d[m].to_numpy(float))})
table = pd.DataFrame(rows).set_index('model')
table.round(4)
"""),
    md("## Bootstrap-CI на Gini и top-10 % capture"),
    code("""ci_rows = []
for m in models:
    for metric_name, fn in [('gini', gini), ('top10_capture', lambda a, b: top_capture(a, b, .10))]:
        r = bootstrap_ci(y, d[m].to_numpy(float), fn, RNG, n_boot=1000)
        ci_rows.append({'model': m, 'metric': metric_name,
                        'point': r['point'], 'lo': r['lo'], 'hi': r['hi'], 'n_boot': 1000})
ci = pd.DataFrame(ci_rows)
ci.pivot(index='model', columns='metric', values=['point', 'lo', 'hi']).round(4)
"""),
    md("## Парные сравнения: BG/NBD против бейзлайнов"),
    code("""cmp_rows = []
for other in ['B1', 'B3', 'B4', 'MBG', 'Pareto']:
    r = bootstrap_diff(y, d.BG.to_numpy(float), d[other].to_numpy(float), gini, RNG, n_boot=1000)
    cmp_rows.append({'comparison': f'BG − {other}', 'metric': 'gini', **r})
    r2 = bootstrap_diff(y, d.BG.to_numpy(float), d[other].to_numpy(float),
                        lambda a, b: top_capture(a, b, .10), RNG, n_boot=1000)
    cmp_rows.append({'comparison': f'BG − {other}', 'metric': 'top10_capture', **r2})
compare = pd.DataFrame(cmp_rows)
compare.round(4)
"""),
    md("""Интерпретация: интервал разницы не пересекает 0 — преимущество есть; пересекает —
разница неотличима от шума. Это и есть правило §6 conventions: метрика, окно,
бейзлайн, интервал."""),
    md("## Калибровка по срезам: когорта, страна, бэнд прогноза"),
    code("""cal_cohort = calibration_by(y, d.BG.to_numpy(), d.cohort.to_numpy())
top_countries = d.country.value_counts().head(6).index
country_group = np.where(d.country.isin(top_countries), d.country, 'other')
cal_country = calibration_by(y, d.BG.to_numpy(), country_group)
band = pd.qcut(d.BG.rank(method='first'), 10, labels=False) + 1
cal_band = calibration_by(y, d.BG.to_numpy(), band.to_numpy())
print('flag по когортам:', int(cal_cohort.flag.sum()), 'из', len(cal_cohort))
print('flag по странам:', int(cal_country.flag.sum()), 'из', len(cal_country))
print('flag по децилям:', int(cal_band.flag.sum()), 'из', len(cal_band))
cal_band.round(3)
"""),
    md("""Calibration ratio = mean(pred)/mean(actual); флаг вне [0.8, 1.2]. Для BG/NBD
уровень систематически выше факта (ratio ≈ 1.35): это ограничение модели на этом
окне, а не ошибка ранжирования."""),
    code("""cal_cohort.to_csv(PROCESSED / 'calibration_by_cohort.csv', index=False)
cal_country.to_csv(PROCESSED / 'calibration_by_country.csv', index=False)
cal_band.to_csv(PROCESSED / 'calibration_by_band.csv', index=False)
"""),
    md("## Графики"),
    code("""plot_lorenz(y, {'B1 историческая сумма': d.B1, 'B3 мультипликатор': d.B3,
                 'BG/NBD + Gamma-Gamma': d.BG},
           FIG / '12-lorenz-holdout.png', title='Holdout: B1, B3 и BG/NBD')
plt.show()
plot_calibration_by_decile(y, d.BG.to_numpy(), FIG / '13-calibration-deciles.png')
plt.show()
"""),
    code("""fig, ax = plt.subplots(figsize=(8, 4.2))
t = table.loc[['B1', 'B3', 'B4', 'BG']].sort_values('gini')
ax.barh(t.index, t.gini, color=['#BAB0AC', '#F58518', '#72B7B2', '#4C78A8'])
ax.set(xlabel='Gini на holdout', title='Ранжирование: модель против бейзлайнов')
fig.tight_layout(); fig.savefig(FIG / '14-model-comparison.png'); plt.show()
"""),
    code("""d[models + ['spend_main']].to_parquet(PROCESSED / 'validation.parquet')
table.to_csv(PROCESSED / 'model_metrics.csv')
ci.to_csv(PROCESSED / 'bootstrap_ci.csv', index=False)
compare.to_csv(PROCESSED / 'model_comparison.csv', index=False)
"""),
    md("""Главный результат: BG/NBD обыгрывает B3 и B4 с интервалом, но B1 остаётся
чуть впереди. Это не провал, а граница применимости: историческая сумма сильна на
12/12-сплите, где размер клиента уже виден; модель добавляет p(alive) и работает
там, где истории мало."""),
]

horizon = [
    md("""# 09 · Горизонт и дисконтирование

Шаг 10 плана. Кривая «ответ vs горизонт» для 6/12/24 месяцев; 6 и 12 наблюдаются,
24 — экстраполяция. Ставка дисконтирования — 10 % годовых, месячная
`(1.10)^(1/12)−1 = 0.797 %`. Формула `m·r/(1+d−r)` вместо `m/churn` — потому что
`m/churn` это тот же геометрический ряд при `d = 0`."""),
    code(HEAD + """
split = pd.read_parquet(PROCESSED / 'ltv_split.parquet')
purch = pd.read_parquet(PROCESSED / 'purchases.parquet')
spend = pd.read_parquet(PROCESSED / 'spend.parquet')
summary = pd.read_csv(PROCESSED / 'bgnbd_summary.csv', index_col=0)
d = split.join(purch).join(spend[['spend_main']])
x = d.frequency.to_numpy(float); t_x = d.recency.to_numpy(float); T = d['T'].to_numpy(float)
params = tuple(float(summary.loc[k, 'mean']) for k in ['r', 'alpha', 'a', 'b'])
"""),
    md("## Кривая «ответ vs горизонт»"),
    code("""rows = []
for H in [6, 12, 24]:
    pred = bgnbd_expected_purchases(params, x, t_x, T, 30 * H) * d.spend_main.to_numpy()
    observed = np.nan
    if H <= 12:
        observed = build_target(tx, d.index, CUTOFF_DATE, H).revenue_holdout.mean()
    rows.append({'horizon': H, 'predicted': pred.mean(), 'observed': observed})
curve = pd.DataFrame(rows)
curve.round(1)
"""),
    code("""plot_horizon(curve, FIG / '15-horizon-curve.png', chosen=12)
plt.show()
"""),
    md("""6 месяцев наблюдается, 12 — тоже; 24 месяца данных нет (окно кончилось
2011-12-09), это экстраполяция модели. 12 выбрано как самый длинный полностью
наблюдаемый горизонт и типичный цикл планирования бюджета."""),
    md("## Дисконтирование"),
    code("""annual_rate = 0.10
d_m = (1 + annual_rate) ** (1 / 12) - 1
monthly_purch = monthly_expected_purchases(bgnbd_expected_purchases, params, x, t_x, T, 12)
monthly_rev = monthly_purch * d.spend_main.to_numpy()[None, :]
pv = discounted_revenue(monthly_rev, annual_rate)
undiscounted = monthly_rev.sum(axis=0)
print('месячная ставка:', round(d_m, 5), f'({d_m:.3%})')
print('средняя 12-мес выручка: undiscounted', round(undiscounted.mean(), 1),
      '£ · PV', round(pv.mean(), 1), '£ · скидка', f'{1 - pv.mean()/undiscounted.mean():.1%}')
print('Gini undiscounted:', round(gini(d.revenue_holdout, undiscounted), 4),
      '· Gini PV:', round(gini(d.revenue_holdout, pv), 4))
"""),
    md("""## Псевдо-holdout: почему нет глобальной поправки

Проверяю, можно ли оценить смещение уровня внутри калибровки: фичу BG/NBD на
более коротком окне и предсказываю следующий период. Если псевдо-holdout даёт
стабильный коэффициент, его можно перенести на основной holdout; если нет —
глобальная поправка не обоснована."""),
    code("""def pseudo_ratio(cutoff, months):
    f = calibration_features(tx, cutoff)
    r = calibration_revenue(tx, cutoff)
    tgt = build_target(tx, f.index, cutoff, months)
    dd = f.join(tgt).join(r.rename('rev_cal'))
    dd = dd[dd['T'] > 0]
    xx = dd.frequency.to_numpy(float); tt = dd.recency.to_numpy(float); TT = dd['T'].to_numpy(float)
    p, _ = fit_bgnbd(xx, tt, TT)
    gg = dd[dd.frequency >= 1]
    pg, _ = fit_gamma_gamma(gg.frequency.to_numpy(float), gg.monetary.to_numpy(float))
    sp = gamma_gamma_conditional_spend(pg, xx, dd.monetary.to_numpy(float))
    pred = bgnbd_expected_purchases(p, xx, tt, TT, 30 * months) * sp
    return len(dd), pred.mean() / dd.revenue_holdout.mean(), gini(dd.revenue_holdout, pred)

pseudo_rows = []
for label, c, h in [('6/6', pd.Timestamp('2010-06-01'), 6), ('9/3', pd.Timestamp('2010-09-01'), 3)]:
    n, ratio, g = pseudo_ratio(c, h)
    pseudo_rows.append({'window': label, 'n': n, 'calibration_ratio': ratio, 'gini': g})
pseudo = pd.DataFrame(pseudo_rows)
pseudo.round(3)
"""),
    md("""Псевдо-holdout даёт calibration ratio 0.86 (6/6) и 0.70 (9/3), а основной
holdout — 1.36. Направление ошибки не переносится: короткое окно недооценивает,
12-месячное переоценивает. Глобальную поправку по калибровке не применяю;
уровень остаётся ограничением, а решение строится на ranking."""),
    md("""## Почему не `m / churn`, а `m·r/(1+d−r)`

`m/churn` — это сумма геометрического ряда `m + m·r + m·r² + …` при `d = 0`.
С дисконтированием каждый член делится на `(1+d)^k`, и сумма равна
`m·r/(1+d−r)`. При `r = 0.96` и `d = 0.797 %` разница с `m/churn` — около 20 %;
при более высоком удержании ошибка растёт."""),
    code("""r = 0.96
m = 1.0
formula = m * r / (1 + d_m - r)
m_over_churn = m / (1 - r)
print('m·r/(1+d−r):', round(formula, 3), '· m/churn:', round(m_over_churn, 3),
      '· разница:', f'{formula/m_over_churn - 1:.1%}')
assert formula < m_over_churn
"""),
    code("""curve.to_csv(PROCESSED / 'horizon_curve.csv', index=False)
pd.DataFrame({'pv': pv, 'undiscounted': undiscounted}, index=d.index).to_parquet(PROCESSED / 'discounted.parquet')
"""),
    md("""Дисконтирование не меняет порядок клиентов (общая кривая скидок), поэтому
ranking-метрики от него не зависят; меняется уровень CLV и решение о CAC."""),
]

segments = [
    md("""# 10 · Сегментация и рекомендация

Шаг 11–12 плана. Порог high-value — число, обоснованное покрытием выручки holdout:
беру минимальный верхний сегмент, который собирает 60 % фактической выручки.
Отдельно — сегмент B (одна покупка) и shrinkage прогноза к среднему по группе
«страна × бэнд первого чека»."""),
    code(HEAD + """
split = pd.read_parquet(PROCESSED / 'ltv_split.parquet')
val = pd.read_parquet(PROCESSED / 'validation.parquet')
d = split.join(val[['BG', 'B1', 'spend_main']])
y = d.revenue_holdout.to_numpy(float)
"""),
    md("## Порог high-value по покрытию 60 % выручки"),
    code("""seg = smallest_segment_for_coverage(y, d.BG.to_numpy(), target=0.60)
print('k:', seg['k'], '· доля клиентов:', f"{seg['share_customers']:.1%}",
      '· покрытие:', f"{seg['coverage']:.1%}", '· порог:', round(seg['threshold'], 1), '£')
threshold = float(np.floor(seg['threshold'] / 10) * 10)
print('порог, округлённый до £10:', threshold, '£')
"""),
    code("""d['high_value'] = d.BG >= threshold
d['one_purchase'] = d.frequency == 0
d['segment'] = np.select([d.high_value, d.one_purchase], ['high-value', 'B (одна покупка)'], default='остальные')
seg_table = d.groupby('segment', observed=True).agg(
    customers=('BG', 'size'),
    customers_share=('BG', lambda s: len(s) / len(d)),
    predicted_ltv_share=('BG', lambda s: s.sum() / d.BG.sum()),
    revenue_share=('revenue_holdout', lambda s: s.sum() / y.sum()),
    mean_actual=('revenue_holdout', 'mean'),
    median_actual=('revenue_holdout', 'median'),
).reset_index()
seg_table.round(3)
"""),
    code("""plot_segments(seg_table, FIG / '16-segments.png')
plt.show()
"""),
    md("""## Сегмент B: shrinkage к группе «страна × бэнд первого чека»

Для клиента с одной покупкой индивидуальный прогноз переобучен. Shrinkage:
`w = n/(n+10)`, где n — число дней с покупкой; для сегмента B w ≈ 0.09, то есть
прогноз почти равен среднему группы. Группа — страна × квартиль первого чека."""),
    code("""first_check = d.spend_main  # для группировки ниже используется первый чек из spend.parquet
spend = pd.read_parquet(PROCESSED / 'spend.parquet')
d['first_check'] = spend['first_check'].reindex(d.index)
d['first_check_band'] = pd.qcut(d.first_check.rank(method='first'), 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
d['group'] = d.country.astype(str) + ' × ' + d.first_check_band.astype(str)
n_obs = (d.frequency + 1).clip(lower=1)
d['BG_shrunk'] = shrink_predictions(d.BG, d.group, n_obs, k=10)
b = d[d.segment == 'B (одна покупка)']
print('B: n', len(b), '· Gini до', round(gini(b.revenue_holdout, b.BG), 4),
      '· после', round(gini(b.revenue_holdout, b.BG_shrunk), 4))
print('B: MAE до', round(mae(b.revenue_holdout, b.BG), 1),
      '· после', round(mae(b.revenue_holdout, b.BG_shrunk), 1))
"""),
    code("""plot_shrinkage(b.BG.to_numpy(), b.BG_shrunk.to_numpy(), b.revenue_holdout.to_numpy(),
               FIG / '17-shrinkage-segment-b.png')
plt.show()
"""),
    md("""Shrinkage меняет не только уровень, но и ranking внутри B: индивидуальный
прогноз одной транзакции шумный, групповое среднее устойчивее."""),
    md("## Рекомендация: цифра и что меняется от действия"),
    code("""hv = d[d.high_value]
median_hv = float(hv.revenue_holdout.median())
margin = 0.30
cac_cap = margin * median_hv
print('high-value: клиентов', len(hv), f'({len(hv)/len(d):.1%})',
      '· покрытие выручки', f'{hv.revenue_holdout.sum()/y.sum():.1%}')
print('медианная фактическая выручка holdout в сегменте:', round(median_hv, 1), '£')
print('CAC cap при марже 30 %:', round(cac_cap, 1), '£ на клиента')
"""),
    code("""recommendation = {
    'threshold_gbp': threshold,
    'segment_share': float(len(hv) / len(d)),
    'revenue_coverage': float(hv.revenue_holdout.sum() / y.sum()),
    'median_holdout_revenue_gbp': median_hv,
    'gross_margin_assumption': margin,
    'cac_cap_gbp': cac_cap,
    'notice': 'рост бюджета на сегмент поднимает CAC; при CAC выше cap сегмент перестаёт окупаться',
}
(PROCESSED / 'recommendation.json').write_text(json.dumps(recommendation, ensure_ascii=False, indent=2), encoding='utf-8')
seg_table.to_csv(PROCESSED / 'segments.csv', index=False)
d[['segment', 'BG', 'BG_shrunk', 'revenue_holdout', 'high_value', 'one_purchase']].to_parquet(PROCESSED / 'segments.parquet')
recommendation
"""),
    md("""**Что меняется от самого действия.** Рост бюджета на high-value сегмент
повышает CAC: аукцион и частота контакта дорожают, а часть сегмента уже купила бы
без стимула. Поэтому рекомендация — не «увеличить бюджет на сегмент», а «держать
CAC ниже £X и перераспределять бюджет только пока предельный CAC ниже cap».
Это общая форма ошибки Альфа-Банка: высокий прогноз дохода не является разрешением
на действие без модели издержек."""),
    md("""Ограничение: маржа 30 % — сценарное допущение, в данных её нет. В README
число помечено как assumption, а не как измеренная величина."""),
]

if __name__ == "__main__":
    run("04-split-calibration-holdout.ipynb", split)
    run("05-baselines.ipynb", baselines)
    run("06-bgnbd-fit.ipynb", bgnbd)
    run("07-gamma-gamma.ipynb", gamma)
    run("08-holdout-validation.ipynb", validation)
    run("09-horizon-discounting.ipynb", horizon)
    run("10-segments-decision.ipynb", segments)
