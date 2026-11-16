# Product Analytics Portfolio

Три кейса: прогноз LTV, A/B-эксперимент и дашборд. Каждый кейс отвечает на бизнес-вопрос, а не пересказывает метод; цифры воспроизводятся ноутбуками, данные и рабочие журналы в репозиторий не входят.

### LTV: прогноз и сегментация клиентской базы

Минимальный верхний сегмент по прогнозному LTV — 9.8 % клиентов при пороге £3 580 — собирает 60.1 % выручки holdout.

- BG/NBD + Gamma-Gamma на 4 239 клиентах, calibration/holdout 12/12 месяцев, holdout наблюдается полностью
- BG/NBD обыгрывает отраслевой мультипликатор B3 на +10.4 Gini-пункта (95 % CI 8.3–12.9), но статистически не отличается от исторической суммы B1 (−0.7 п.п., 95 % CI −1.4…0.02)
- High-value: 9.8 % клиентов, 60.1 % выручки holdout, CAC cap £1 284 при марже 30 %; уровень модели переоценён на 36 %, поэтому рекомендация — по ранжированию

[Разбор кейса →](case_ltv/README.md)

### A/B: дизайн теста и анализ Cookie Cats

`gate_40` не раскатывать: 7-дневный retention ниже на 0.82 п.п. (95 % CI −1.33…−0.31), Holm p = 0.0062; SRM p = 0.0086, но эффект устойчив к downsampling.

- 90 189 игроков, 44 700 `gate_30` / 45 489 `gate_40`; A/A FPR 5.4 % и 6.0 %, SRM — предупреждение, не жёсткая остановка
- Part 1: high-value сегмент LTV — 417 клиентов, baseline repeat90 76.3 %, MDE 11.7 п.п.; для 5 п.п. нужно 1 137/arm, классический тест на сегменте нереалистичен
- Решение: не раскатывать `gate_40`; для LTV-механики расширить сегмент до top 30–50 % или использовать matched control / switchback

[Разбор кейса →](case_ab/README.md)

### Дашборд: юнит-экономика каналов

`paid_search` не окупается: LTV:CAC 0.97 и payback 12.3 месяца; перекладывание 20 % бюджета в email/referral даёт +£50.8 тыс. contribution (+8.7 %), пока эластичность CAC ниже 0.8.

- 3 288 клиентов 2010-когорт, 5 синтетических каналов; CAC и margin — сценарные допущения
- Base: contribution £585 тыс., LTV:CAC 2.27, payback 5.3 мес; optimistic £1.04 млн, pessimistic £365 тыс.
- SQLite-витрина, Streamlit-приложение и 5 скриншотов; публичная ссылка — после деплоя

[Разбор кейса →](case_dashboard/README.md)

---

## Структура

```
case_ltv/            готовый кейс: src, notebooks 01–10, figures, README
case_ab/             скелет A/B-кейса (в работе)
case_dashboard/      скелет дашборда (в работе)
docs/                снапшоты conventions.md и datasets.md
tests/               unit-тесты формул и метрик без данных
paths.py             единая точка путей к data/ и library/ вне репозитория
sync_check.py        аудит синхронизации и гигиены
```

## Разработка

```bash
pip install -r requirements-dev.txt
python -m pytest
python -m ruff check case_ltv/src tests
python sync_check.py --audit
```

`requirements.txt` — runtime для ноутбуков, `requirements-dev.txt` — тесты и линтер. CI в `.github/workflows/ci.yml` запускает тесты, ruff и compileall; ноутбуки в CI не исполняются, потому что для них нужен датасет. Правила для людей и агентов — в [CONTRIBUTING.md](CONTRIBUTING.md) и [AGENT.md](AGENT.md); стандарт кейса — в [docs/conventions.md](docs/conventions.md).

Данные: Online Retail II (UCI), CC BY 4.0, Chen, D. (2012), DOI 10.24432/C5CG6D. Данные, библиотека источников и рабочие журналы лежат вне репозитория; в git едут код, ноутбуки с выводом, графики, тесты и README.
