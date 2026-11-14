"""Собирает и исполняет notebooks/01-load-and-profile.ipynb (Restart & Run All)."""

from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

NB = Path(__file__).resolve().parents[1] / "notebooks" / "01-load-and-profile.ipynb"

md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

cells = [
    md("# 01 · Загрузка и профилирование Online Retail II\n\n"
       "Шаги 1–2 плана `case_ltv`. Источник: Chen, D. (2012), UCI ML Repository, "
       "DOI 10.24432/C5CG6D, CC BY 4.0. Чистка и её журнал — в `src/data.py`."),
    code("import sys\n"
         "sys.path.insert(0, '../..')      # корень репо: paths.py\n"
         "sys.path.insert(0, '../src')\n"
         "import numpy as np, pandas as pd, matplotlib.pyplot as plt\n"
         "from paths import CASE_LTV, figures\n"
         "from data import load_raw, load_clean\n"
         "FIG = figures(CASE_LTV); FIG.mkdir(parents=True, exist_ok=True)\n"
         "raw = load_raw()\n"
         "tx, log = load_clean()\n"
         "print(raw.shape, tx.shape)"),
    md("## Пропуски в сырых данных"),
    code("na = raw.isna().mean().round(4).rename('null_share').to_frame()\n"
         "na"),
    md("Гейт из журнала: доля пропусков в `invoice`, `date`, `price` < 20 %. "
       "`customer_id` пропущен у ~23 %, строки удаляются, доля фиксируется ниже."),
    code("assert (na.loc[['invoice', 'date', 'price'], 'null_share'] < 0.20).all()\n"
         "print('null customer_id, доля сырых строк:', round(na.loc['customer_id', 'null_share'], 4))"),
    md("## Журнал чистки\n\nКаждая строка — шаг по порядку. Порядок влияет на числа "
       "(дубликаты удаляются до пропусков ID)."),
    code("log"),
    md("Отмены (счета с префиксом `C`) не удаляются, остаются отрицательными строками: "
       "нужны для net-выручки. Служебные коды (`POST`, `DOT`, `M`, `D`, `C2`, `BANK CHARGES`, "
       "`AMAZONFEE`, подарочные ваучеры `gift_*` и др.) — не покупка товара клиентом."),
    code("print('период:', tx.date.min(), '→', tx.date.max())\n"
         "months = (tx.date.max() - tx.date.min()).days / 30.44\n"
         "n_cust = tx.customer_id.nunique()\n"
         "print(f'история {months:.1f} мес, клиентов {n_cust}, строк {len(tx)}')\n"
         "assert n_cust >= 1000 and months >= 18      # гейты качества данных"),
    md("## Отмены и возвраты"),
    code("sales = tx[~tx.is_cancel]; canc = tx[tx.is_cancel]\n"
         "print('строк отмен:', len(canc), f'({len(canc)/len(tx):.1%})')\n"
         "print('выручка продаж:', round(sales.revenue.sum()), ' отмен:', round(canc.revenue.sum()),\n"
         "      f' доля отмен: {-canc.revenue.sum()/sales.revenue.sum():.1%}')\n"
         "# возврат без исходной покупки в данных = покупка до начала окна\n"
         "first = sales.groupby('customer_id').date.min()\n"
         "early = canc.assign(first=canc.customer_id.map(first)).query('date < first')\n"
         "print('отмен раньше первой покупки клиента:', len(early))"),
    md("## Крупнейшие строки (выбросы)"),
    code("pd.concat([tx.nlargest(5, 'revenue'), tx.nsmallest(5, 'revenue')])"
         "[['invoice','stock_code','quantity','price','revenue','customer_id','date']]"),
    md("Две пары «заказ — полная отмена»: клиент 16446 (80 995 шт., ±168 469,6 £, "
       "обе строки 2011-12-09) и клиент 12346 (74 215 шт., ±77 183,6 £, отмена в тот же "
       "месяц). Нетто 16446 равно нулю, у 12346 остаётся его обычная история. "
       "Фильтр «отбросить отрицательные» оставил бы только положительную половину "
       "и завысил бы выручку обоих клиентов."),
    code("big = tx.loc[tx.revenue.abs() > 100_000]\n"
         "print(big.groupby('customer_id').revenue.sum())"),
    md("## Распределения: покупки на клиента и размер чека\n\n"
       "Покупка = уникальный `invoice` без префикса `C`."),
    code("inv = sales.groupby(['customer_id', 'invoice']).agg(date=('date','min'), amount=('revenue','sum')).reset_index()\n"
         "n_purch = inv.groupby('customer_id').size()\n"
         "print(n_purch.describe())\n"
         "print('ровно 1 покупка:', f'{(n_purch==1).mean():.1%}', ' 2+:', f'{(n_purch>=2).mean():.1%}')\n"
         "fig, ax = plt.subplots(1, 2, figsize=(11, 3.8), dpi=150)\n"
         "ax[0].hist(n_purch.clip(upper=30), bins=30)\n"
         "ax[0].set(xlabel='покупок на клиента (обрезано на 30)', ylabel='клиентов', title='Частота покупок за весь период')\n"
         "ax[1].hist(np.log10(inv.amount.clip(lower=1)), bins=50)\n"
         "ax[1].set(xlabel='log10 суммы чека, £', ylabel='чеков', title='Размер чека')\n"
         "fig.tight_layout(); fig.savefig(FIG / '01-purchases-and-basket.png'); plt.show()"),
    md("Слева: доля клиентов с одной покупкой — размер сегмента B (журнал решений, "
       "2026-09-13), справа: распределение чека скошено вправо, оптимально в логарифме."),
    md("## Активность во времени"),
    code("m = inv.set_index('date').resample('MS').agg(orders=('amount','size'), revenue=('amount','sum'))\n"
         "fig, ax = plt.subplots(figsize=(10, 3.6), dpi=150)\n"
         "ax.plot(m.index, m.revenue / 1e3)\n"
         "ax.set(ylabel='выручка нетто чеков, тыс. £', title='Месячная выручка (покупки без отмен)')\n"
         "fig.tight_layout(); fig.savefig(FIG / '02-monthly-revenue.png'); plt.show()\n"
         "m.round(0).T"),
    md("Максимум в ноябре обоих лет (2010-11: 1,16 млн £, 2011-11: 1,14 млн £) — сезонность "
       "подарочного ритейла. Декабрь 2011 неполный (данные до 9-го числа), поэтому "
       "падение в конце ряда не является оттоком. Cutoff для calibration/holdout "
       "выбираем с учётом обоих эффектов."),
    code("# Состав по странам\n"
         "c = tx.groupby('country').agg(customers=('customer_id','nunique'), revenue=('revenue','sum'))\n"
         "c['rev_share'] = (c.revenue / c.revenue.sum()).round(3)\n"
         "c.sort_values('revenue', ascending=False).head(6)"),
]

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
NotebookClient(nb, timeout=600, kernel_name="python3",
               resources={"metadata": {"path": str(NB.parent)}}).execute()
nbf.write(nb, NB)
print("written", NB)
