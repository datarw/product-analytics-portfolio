"""Собирает notebooks/12-effect-and-guardrails.ipynb."""

from pathlib import Path

import nbformat as nbf

NBDIR = Path(__file__).resolve().parents[1] / "notebooks"
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

HEAD = """import sys
sys.path.insert(0, '../..'); sys.path.insert(0, '../src')
import json
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from paths import CASE_AB, PROCESSED, RANDOM_SEED, figures
from data import load_clean
from experiment import (srm_test, baseline_table, guardrail_table,
                        downsample_diff, holm_adjust)
FIG = figures(CASE_AB); FIG.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(RANDOM_SEED)
df = load_clean()
"""

cells = [
    md("""# 12 · Cookie Cats: эффект, guardrails и решение

Шаг 14–15 и 18 плана. Сначала проверяю, не создан ли эффект SRM: многократно
уменьшаю перевешенную группу до размера меньшей. Затем считаю эффект на
retention, guardrail на игровые раунды и Holm-поправку на множественные метрики.
Решение — предварительное, до README и Part 1."""),
    code(HEAD + "print('n =', len(df), '· groups =', df.version.value_counts().to_dict())"),
    md("""## SRM-чувствительность

SRM p = 0.0086 — предупреждение. Если эффект создан лишними 789 пользователями
в `gate_40`, уменьшение группы до 44 700 должно его убрать. Если эффект
сохраняется — SRM объясняет дисперсию, но не направление."""),
    code("""sens_rows = []
for metric in ['retention_1', 'retention_7']:
    sens_rows.append(downsample_diff(df, metric, RNG, n_iter=1000))
sens = pd.DataFrame(sens_rows)
sens.round(5)
"""),
    md("""## Эффект и guardrails

`retention_1` и `retention_7` — основные метрики. Guardrails: доля нулевых
раундов и медиана `sum_gamerounds`; распределение тяжёлое, поэтому медиана и
Mann–Whitney, а не среднее."""),
    code("""guard = guardrail_table(df, RNG)
guard.round(5)
"""),
    code("""effect_per_million = 1_000_000 * guard.loc[guard.metric == 'retention_7', 'diff'].iloc[0]
print('эффект retention_7 на 1 млн установок:', round(effect_per_million))
"""),
    md("""## Лесной график: размер эффекта и интервал

Красным — значимо после Holm-поправки на четыре метрики. Нулевая линия —
отсутствие эффекта."""),
    code("""prop = guard[guard.metric.isin(['retention_1', 'retention_7', 'zero_rounds'])].reset_index(drop=True)
fig, ax = plt.subplots(figsize=(7, 3.6), dpi=150)
for i, row in prop.iterrows():
    color = '#E45756' if row.p_holm < 0.05 else '#4C78A8'
    ax.errorbar(row['diff'] * 100, i,
                xerr=[[(row['diff'] - row['ci_low']) * 100],
                      [(row['ci_high'] - row['diff']) * 100]],
                fmt='o', capsize=5, color=color)
ax.axvline(0, color='black', ls='--', lw=1)
ax.set_yticks(range(len(prop)), prop.metric)
ax.set(xlabel='разница gate_40 − gate_30, п.п.', title='Эффект с 95 % CI; красный — значим после Holm')
fig.tight_layout(); fig.savefig(FIG / '04-effect-forest.png'); plt.show()
"""),
    md("""## Guardrail: распределение раундов"""),
    code("""fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
for version in ['gate_30', 'gate_40']:
    x = np.sort(df.loc[df.version == version, 'sum_gamerounds'].to_numpy())
    y = np.arange(1, len(x) + 1) / len(x)
    ax.plot(np.log1p(x), y, lw=1.5, label=version)
ax.set(xlabel='log1p(sum_gamerounds)', ylabel='ECDF',
       title='Распределение игровых раундов за 14 дней')
ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(FIG / '05-gamerounds-ecdf.png'); plt.show()
"""),
    md("""## Решение

- `retention_7`: −0.82 п.п. (95 % CI −1.33…−0.31), p = 0.0016, Holm p = 0.0062.
  Направление отрицательное и устойчиво к SRM: после downsampling −0.82 п.п.
  (CI −0.87…−0.78).
- `retention_1`: −0.59 п.п. (−1.24…+0.06), p = 0.074, Holm p = 0.151 — не значимо.
- Guardrails: доля нулевых раундов +0.19 п.п. (p = 0.169); медиана раундов
  17 против 16, Mann–Whitney p = 0.050, Holm p = 0.151 — значимого выигрыша нет.
- **Не раскатывать `gate_40`.** 7-дневный retention хуже, 1-дневный не лучше,
  engagement не выигрывает. Перед повторным тестом — разобраться в SRM:
  логирование, дедупликация, фактическая вероятность назначения.
- Практическая значимость: −0.82 п.п. на 1 млн установок — около 8 200
  не вернувшихся на 7-й день; денежного эффекта в данных нет, поэтому это
  retention-решение, а не финансовая оценка."""),
    code("""summary = {
    'srm_sensitivity': sens.to_dict(orient='records'),
    'guardrails': guard.to_dict(orient='records'),
    'decision': 'do_not_rollout_gate_40',
    'reason': 'retention_7 negative and Holm-significant; no guardrail gain; SRM warning',
}
(PROCESSED / 'cookie_cats_effect.json').write_text(
    json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
print('saved', PROCESSED / 'cookie_cats_effect.json')
"""),
]

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
NBDIR.mkdir(parents=True, exist_ok=True)
nbf.write(nb, NBDIR / "12-effect-and-guardrails.ipynb")
print("written", NBDIR / "12-effect-and-guardrails.ipynb")
