# Product Analytics Portfolio

Публичная витрина: дашборд юнит-экономики каналов на данных Online Retail II. Полные кейсы LTV и A/B и код в публичном репозитории не опубликованы.

**[Открыть дашборд →](https://datarw.github.io/product-analytics-portfolio/case_dashboard/reports/dashboard.html)**

## Что на дашборде

- North Star: 12-month contribution margin per acquired customer = gross-profit LTV − CAC.
- 3 288 клиентов 2010-когорт, 5 синтетических каналов; CAC и margin — сценарные допущения.
- Base: contribution £585 тыс., LTV:CAC 2.27, payback 5.3 мес.
- `paid_search`: LTV:CAC 0.97, payback 12.3 мес → pause; `email`, `referral`, `organic` → scale.
- Перекладывание 20 % бюджета keep-каналов: +£50.8 тыс. contribution (+8.7 %) при эластичности CAC 0.5; break-even — 0.8.

## Данные и ограничения

Online Retail II (UCI), CC BY 4.0, Chen, D. (2012), DOI 10.24432/C5CG6D. Данные в репозиторий не входят. Каналы синтетические; CAC и margin — сценарные допущения, а не фактические данные. Дашборд показывает метод принятия решения, а не эффективность реальных каналов.

## Что ещё было в работе

LTV-кейс (BG/NBD + Gamma-Gamma, calibration/holdout 12/12) и A/B-кейс (Cookie Cats, SRM/A/A, решение не раскатывать `gate_40`) — полные тексты и код не входят в публичную витрину.
