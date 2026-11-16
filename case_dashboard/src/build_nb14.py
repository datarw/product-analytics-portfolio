"""Собирает notebooks/14-unit-economics.ipynb."""

from pathlib import Path

import nbformat as nbf

NBDIR = Path(__file__).resolve().parents[1] / "notebooks"
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

HEAD = """import sys
sys.path.insert(0, '../..'); sys.path.insert(0, '../src')
import pandas as pd
from IPython.display import Image, display
from paths import CASE_DASHBOARD, PROCESSED
from build_mart import build
base, ch, scen, realloc_sum = build()
"""

cells = [
    md("""# 14 · Unit economics: каналы, сценарии и перераспределение

Шаги 3–6 и 8 плана. Считаю unit economics на 2010-когортах LTV-базы:
gross-profit LTV, CAC по синтетическим каналам, contribution, LTV:CAC,
payback. Затем — сценарии и перераспределение бюджета с ростом CAC."""),
    code(HEAD + "print('клиентов:', len(base), '· каналов:', base.channel.nunique())\nprint('LTV:CAC:', round(base.ltv_gp.sum()/base.cac.sum(), 3), '· contribution:', round(base.contribution.sum()))"),
    md("""## Каналы

Каналы синтетические: softmax по первому чеку, CAC — сценарное допущение.
Organic не масштабируется бюджетом; в таблице он остаётся для полноты."""),
    code("ch.round(3)"),
    md("""## Сценарии

Optimistic: LTV без поправки, CAC ×0.85. Base: LTV ×1/1.364 (калибровка LTV-кейса),
CAC базовый. Pessimistic: LTV ×0.9/1.364, CAC ×1.25."""),
    code("scen.round(3)"),
    md("""## Перераспределение и эластичность CAC

20 % бюджета keep-каналов перекладывается в scale-каналы. При эластичности 0.5
CAC получающих каналов растёт; при 1.0 перекладывание уже убыточно. Break-even
эластичность — в meta."""),
    code("print('reallocation 20% / 0.5:', {k: round(v, 3) if isinstance(v, float) else v for k, v in realloc_sum.items()})\ngrid = pd.read_csv(PROCESSED / 'dashboard_reallocation_grid.csv')\ngrid.pivot(index='shift', columns='elasticity', values='uplift_pct').round(3)"),
    md("""## Скриншоты дашборда"""),
    code("""for name in ['01-dashboard-overview', '02-channel-unit-economics', '03-cohort-contribution',
             '04-scenarios', '05-reallocation']:
    display(Image(filename=str(CASE_DASHBOARD / 'reports' / 'figures' / f'{name}.png')))
"""),
    md("""## Вывод

- Base: contribution £585 тыс., LTV:CAC 2.27, payback 5.3 мес.
- `paid_search`: LTV:CAC 0.97, payback 12.3 мес → pause.
- `social`: 1.40 / 8.6 мес → keep.
- `email`, `referral`, `organic`: 15.4 / 10.7 / 6.5 → scale.
- 20 % shift при эластичности 0.5 даёт +£50.8 тыс. (+8.7 %); при эластичности
  ≥0.8 перекладывание перестаёт окупаться.
"""),
]

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
NBDIR.mkdir(parents=True, exist_ok=True)
nbf.write(nb, NBDIR / "14-unit-economics.ipynb")
print("written", NBDIR / "14-unit-economics.ipynb")
