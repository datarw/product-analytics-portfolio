-- Витрина дашборда. Таблица unit_economics создаётся build_mart.py.
-- Поля:
--   customer_id      клиент (ключ)
--   cohort           месяц первой покупки
--   country          страна
--   channel          синтетический канал привлечения (сценарная надстройка)
--   first_check      первый чек, £
--   pv               прогнозный 12-месячный PV из LTV-кейса, £
--   ltv_gp           gross-profit LTV = pv * ltv_factor * 0.30, £
--   cac              сценарный CAC канала, £
--   contribution     ltv_gp - cac, £
--   ltv_cac          ltv_gp / cac
--   payback_months   cac / (ltv_gp / 12)
--   decision         pause / keep / scale по порогам

DROP VIEW IF EXISTS channel_summary;
CREATE VIEW channel_summary AS
SELECT
    channel,
    COUNT(*) AS n,
    SUM(cac) AS budget,
    SUM(ltv_gp) AS ltv_gp,
    SUM(contribution) AS contribution,
    CASE WHEN SUM(cac) > 0 THEN SUM(ltv_gp) / SUM(cac) END AS ltv_cac,
    CASE WHEN SUM(ltv_gp) > 0 THEN SUM(cac) / (SUM(ltv_gp) / 12.0) END AS payback_months,
    AVG(ltv_gp) AS ltv_gp_mean,
    AVG(cac) AS cac_mean
FROM unit_economics
GROUP BY channel;

DROP VIEW IF EXISTS cohort_summary;
CREATE VIEW cohort_summary AS
SELECT
    cohort,
    channel,
    COUNT(*) AS n,
    SUM(cac) AS budget,
    SUM(ltv_gp) AS ltv_gp,
    SUM(contribution) AS contribution,
    CASE WHEN SUM(cac) > 0 THEN SUM(ltv_gp) / SUM(cac) END AS ltv_cac,
    CASE WHEN SUM(ltv_gp) > 0 THEN SUM(cac) / (SUM(ltv_gp) / 12.0) END AS payback_months
FROM unit_economics
GROUP BY cohort, channel;

DROP VIEW IF EXISTS cohort_totals;
CREATE VIEW cohort_totals AS
SELECT
    cohort,
    COUNT(*) AS n,
    SUM(cac) AS budget,
    SUM(ltv_gp) AS ltv_gp,
    SUM(contribution) AS contribution,
    CASE WHEN SUM(cac) > 0 THEN SUM(ltv_gp) / SUM(cac) END AS ltv_cac
FROM unit_economics
GROUP BY cohort;

DROP VIEW IF EXISTS decision_counts;
CREATE VIEW decision_counts AS
SELECT decision, COUNT(*) AS n, SUM(cac) AS budget, SUM(contribution) AS contribution
FROM unit_economics
GROUP BY decision;
