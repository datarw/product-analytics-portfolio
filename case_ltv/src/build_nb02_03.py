"""Собирает и исполняет notebooks/02-rfm-table.ipynb и 03-cohorts-retention.ipynb."""

from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

NBDIR = Path(__file__).resolve().parents[1] / "notebooks"
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

HEAD = ("import sys\n"
        "sys.path.insert(0, '../..'); sys.path.insert(0, '../src')\n"
        "import numpy as np, pandas as pd, matplotlib.pyplot as plt\n"
        "from paths import CASE_LTV, figures\n"
        "from data import CLEAN_PARQUET\n"
        "from features import build_features, cohort_tables, calendar_retention\n"
        "FIG = figures(CASE_LTV); FIG.mkdir(parents=True, exist_ok=True)\n"
        "tx = pd.read_parquet(CLEAN_PARQUET)\n"
        "feats = build_features(tx)\n")


def run(name: str, cells: list) -> None:
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    NotebookClient(nb, timeout=600, kernel_name="python3",
                   resources={"metadata": {"path": str(NBDIR)}}).execute()
    nbf.write(nb, NBDIR / name)
    print("written", name)


rfm = [
    md("# 02 · RFM-таблица\n\nШаг 3 плана. Таблица на клиента по конвенции `lifetimes`; "
       "определения — в docstring `src/features.py`. Данные: вся история, cutoff — конец данных."),
    code(HEAD + "feats.describe().T[['count','mean','50%','max']].round(1)"),
    md("## Проверки целостности\n\n"
       "`frequency` пересобирается независимо из сырой таблицы (правило 2 из `conventions.md` §5а): "
       "число уникальных дней с покупкой минус один."),
    code("raw_sales = tx[~tx.is_cancel]\n"
         "freq_check = raw_sales.groupby('customer_id').date.apply(lambda s: s.dt.normalize().nunique()) - 1\n"
         "assert (freq_check.reindex(feats.index) == feats.frequency).all()\n"
         "assert (feats['T'] >= feats.recency).all()\n"
         "assert (feats['T'] - feats.recency == feats.recency_rfm).all()\n"
         "# monetary не определён ровно там, где нет повторных покупок: NaN, а не 0\n"
         "assert (feats.monetary.isna() == (feats.frequency == 0)).all()\n"
         "assert (feats.monetary.dropna() > 0).all()\n"
         "print('ok; клиентов', len(feats), '· frequency = 0:', f'{(feats.frequency == 0).mean():.1%}')"),
    md("Доля `frequency = 0` здесь (дни с покупкой) равна 28,6 %, а доля клиентов с "
       "одним счётом — 27,6 % (ноутбук 01): часть клиентов делала несколько счетов в один день. "
       "Для BG/NBD единицей служит день, поэтому берём 28,6 %."),
    md("## Распределения"),
    code("fig, ax = plt.subplots(1, 3, figsize=(14, 3.8), dpi=150)\n"
         "ax[0].hist(feats.frequency.clip(upper=40), bins=40)\n"
         "ax[0].set(xlabel='frequency (обрезано на 40), повторных дней', ylabel='клиентов', title='Frequency')\n"
         "ax[1].scatter(feats['T'], feats.recency, s=3, alpha=.3)\n"
         "ax[1].plot([0, 730], [0, 730], 'k--', lw=1)\n"
         "ax[1].set(xlabel='T, дней в базе', ylabel='recency, дней', title='Recency против T (ниже диагонали)')\n"
         "ax[2].hist(np.log10(feats.monetary.dropna()), bins=50)\n"
         "ax[2].set(xlabel='log10 средней суммы повторной покупки, £', ylabel='клиентов', title='Monetary (frequency ≥ 1)')\n"
         "fig.tight_layout(); fig.savefig(FIG / '03-rfm-distributions.png'); plt.show()"),
    md("Слева: тяжёлый правый хвост частоты, 1 673 клиента с `frequency = 0`. В центре: "
       "горизонтальная полоса на `recency = 0` — клиенты с одним днём покупки; вертикальная "
       "полоса у `T ≈ 730` — клиенты с первой покупкой в декабре 2009 (см. ноутбук 03); "
       "точки у диагонали — клиенты, чья последняя покупка близка к cutoff. Справа: "
       "monetary в логарифме близок к симметричному (форма похожа на лог-нормальную, "
       "формальный тест не проводился)."),
    md("## Концентрация выручки\n\nГрубая оценка потолка сегментации: какая доля net-выручки "
       "приходится на верх базы (по факту всей истории, не по прогнозу)."),
    code("v = feats.net_revenue.sort_values(ascending=False)\n"
         "share = v.cumsum() / v.sum()\n"
         "for q in (.01, .05, .10, .20):\n"
         "    print(f'топ-{q:.0%} клиентов: {share.iloc[int(len(v)*q)-1]:.1%} net-выручки')\n"
         "print('клиентов с отрицательной net-выручкой:', (feats.net_revenue < 0).sum())\n"
         "x = np.arange(1, len(v)+1) / len(v)\n"
         "fig, ax = plt.subplots(figsize=(5, 4.2), dpi=150)\n"
         "ax.plot(x, share.values, label='факт, вся история'); ax.plot([0,1],[0,1],'k--',lw=1,label='равенство')\n"
         "ax.set(xlabel='доля клиентов (по убыванию выручки)', ylabel='доля net-выручки', title='Концентрация выручки')\n"
         "ax.legend(); fig.tight_layout(); fig.savefig(FIG / '04-revenue-concentration.png'); plt.show()"),
    md("Топ-20 % по факту всей истории собирают 76,8 % net-выручки, топ-10 % — 63,2 %. "
       "Это верхняя граница для сегментации по прогнозу на тех же клиентах и тот же период: "
       "прогноз не может собрать больше, чем собирает сам факт. Для holdout (шаг 9) "
       "аналогичная граница считается отдельно."),
    code("feats.to_parquet(__import__('paths').PROCESSED / 'rfm_full.parquet')"),
]

cohorts = [
    md("# 03 · Когорты и retention\n\nШаг 4 плана. Когорта — месяц **первой покупки**, возраст "
       "считается от **даты** первой покупки блоками по 30 дней (журнал 2026-09-15, Аникин). "
       "Ячейка без полного окна наблюдения — NaN, а не 0."),
    code(HEAD + "size, ret, rpc = cohort_tables(tx, feats)\nsize.to_frame().T"),
    md("## Когорта 2009-12 не является когортой привлечения\n\n"
       "Окно данных начинается 2009-12-01, поэтому «первая покупка» старых клиентов в декабре "
       "2009 — просто первая покупка в окне. Размер когорты это показывает."),
    code("print('медиана размера других когорт:', int(size.iloc[1:].median()), ' · 2009-12:', int(size.iloc[0]))\n"
         "pd.DataFrame({'ret_age1': ret[1], 'ret_age6': ret[6]}).round(3).iloc[[0,1,2,3,4,5]]"),
    md("Когорта 2009-12 (951 клиент) в 5,1 раза крупнее медианной (186) и удерживается лучше "
       "(36,7 % на возрасте 1 против 15–31 % у пяти следующих когорт). Это смесь новых и "
       "давних клиентов. Из кривых привлечения она исключается; "
       "в фите BG/NBD клиенты остаются: их `T` посчитан от первой покупки в окне, "
       "что оговаривается в README."),
    code("import matplotlib as mpl\n"
         "use = ret.iloc[1:]\n"
         "fig, ax = plt.subplots(figsize=(9, 4.4), dpi=150)\n"
         "cmap = plt.get_cmap('viridis')\n"
         "for i, (c, row) in enumerate(use.iterrows()):\n"
         "    ax.plot(row.index, row.values, color=cmap(i / len(use)), lw=1.2, alpha=.8)\n"
         "ax.set(xlabel='возраст, 30-дневные блоки после первой покупки', ylabel='доля клиента когорты с покупкой',\n"
         "       title='Retention по когортам первой покупки (2010-01 … , без 2009-12)')\n"
         "sm = mpl.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(0, len(use)-1))\n"
         "fig.colorbar(sm, ax=ax, label='номер когорты (0 = 2010-01)')\n"
         "fig.tight_layout(); fig.savefig(FIG / '05-cohort-retention.png'); plt.show()"),
    md("Каждая линия — когорта; обрыв линии — граница наблюдаемости. После падения "
       "до 21,6 % на возрасте 1 доля активных убывает медленно (см. взвешенную таблицу "
       "ниже: 17,6 % на возрасте 6, 12,9 % на 10) и не геометрически. Это доля активных "
       "в блоке, а не выживание: клиент может пропустить блок и вернуться. Возврат "
       "кривых вверх на возрасте 11–12 — гипотеза: сезонность (поздняя осень); не проверялось, "
       "и на этих возрастах наблюдаются только ранние когорты."),
    code("pooled = []\n"
         "for k in range(13):\n"
         "    ok = use[k].notna()\n"
         "    n = size.iloc[1:][ok]\n"
         "    pooled.append((use[k][ok] * n).sum() / n.sum() if ok.any() else np.nan)\n"
         "pooled = pd.Series(pooled, name='retention, взвешенно по размеру когорт')\n"
         "pooled.round(3).to_frame().T"),
    md("## Revenue на клиента по возрасту (накопленная кривая)"),
    code("cum = rpc.iloc[1:].cumsum(axis=1, skipna=False)\n"
         "fig, ax = plt.subplots(figsize=(9, 4.4), dpi=150)\n"
         "for i, (c, row) in enumerate(cum.iterrows()):\n"
         "    ax.plot(row.index, row.values, color=cmap(i / len(cum)), lw=1.2, alpha=.8)\n"
         "ax.set(xlabel='возраст, 30-дневные блоки', ylabel='накопленная выручка продаж на клиента когорты, £',\n"
         "       title='Накопленная выручка на клиента по когортам')\n"
         "fig.tight_layout(); fig.savefig(FIG / '06-cohort-cumulative-revenue.png'); plt.show()\n"
         "last = cum.apply(lambda r: r.dropna().iloc[-1]); print('медиана когорт на последней наблюдаемой точке, £:', round(last.median()))"),
    md("Кривая используется как основа для бейзлайна B4 (сумма когортной кривой) и для "
       "выбора точки `t` бейзлайна B3 (шаг 6). Разброс между когортами по накопленной "
       "выручке — прямая оценка того, насколько среднее по когорте (B1) «не знает» "
       "о конкретной когорте."),
    md("## Возраст от даты против календарного месяца\n\n"
       "Сравнение на одних и тех же когортах: доля вернувшихся в возрасте 1 "
       "(30–59 дней после первой покупки) против доли купивших в следующем календарном месяце."),
    code("cal = calendar_retention(tx, feats)\n"
         "cmp_ = pd.DataFrame({'age_aligned': ret[1], 'calendar_next_month': cal[1]}).dropna()\n"
         "cmp_['diff'] = cmp_.calendar_next_month - cmp_.age_aligned\n"
         "print(cmp_.iloc[1:].round(3).describe().loc[['count','mean','std','min','max']])\n"
         "first = feats.first_purchase\n"
         "dom = first.dt.day\n"
         "print('доля клиентов, чья первая покупка приходится на последнюю неделю месяца (дни 24+):', f'{(dom >= 24).mean():.1%}')"),
    md("Смысл ошибки календарной когорты: клиент, купивший 28-го, получает в «следующем "
       "месяце» окно в 31 день после первой покупки, а купивший 2-го — до 60. Окна "
       "в среднем сравнимы, но разные для разных клиентов, и при экстраполяции `p^N` "
       "эта вариация возводится в степень. На возрасте 1 (21 когорта, парное сравнение) "
       "календарный вариант ниже на 1,0 п.п., стандартное отклонение разностей 3,8 п.п., "
       "то есть разница неотличима от шума. На возрастах 4–8 календарные доли выше "
       "на 0,8–1,7 п.п. (средние по когортам, не парные: наборы наблюдаемых когорт "
       "различаются; интервалов нет). Вывод осторожный: эффект в этих данных мал, "
       "в README заявляется метод (возраст от даты), а не выигрыш от него."),
    code("pd.DataFrame({'age_aligned': ret.iloc[1:, :9].mean().round(3), 'calendar': cal.iloc[1:, :9].mean().round(3)}).T"),
]

if __name__ == "__main__":
    run("02-rfm-table.ipynb", rfm)
    run("03-cohorts-retention.ipynb", cohorts)
