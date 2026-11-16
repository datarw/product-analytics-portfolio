# Contributing

Репозиторий — портфолио, но код и ноутбуки должны воспроизводиться с чистого клона. Правки приветствуются через issue или pull request.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/macOS
pip install -r requirements-dev.txt
```

`requirements.txt` — runtime для ноутбуков; `requirements-dev.txt` добавляет `pytest` и `ruff`.

## Data

Данные не входят в репозиторий. Скачайте `online_retail_II.xlsx` со страницы UCI (DOI 10.24432/C5CG6D) и положите рядом с репозиторием:

```
Product_Analytics_Portfolio/
├── product-analytics-portfolio/
└── data/raw/online-retail-ii/online_retail_II.xlsx
```

Затем:

```bash
python case_ltv/src/data.py
python case_ltv/src/execute_notebooks.py
```

На Windows: `$env:PYTHONIOENCODING = 'utf-8'`.

## Style

- Python 3.12, PEP 8, длина строки до 100 символов.
- Пути только через `paths.py`.
- Один `numpy.random.Generator` из `RANDOM_SEED = 42`.
- Ноутбуки — по одному логическому шагу, с сохранённым выводом.
- Числа в README — только из ячеек ноутбуков.
- Новое решение — сначала в рабочий журнал, потом в код.

## Tests

```bash
python -m pytest
python -m ruff check case_ltv/src tests
python -m compileall -q case_ltv/src
```

Тесты не требуют данных: они проверяют формулы и метрики на синтетических примерах. Ноутбуки в CI не исполняются, потому что для них нужен датасет.

## Pull requests

1. Ветка от `main`: `case/<name>` или `chore/<name>`.
2. Один шаг — один осмысленный коммит.
3. Перед PR: тесты, ruff, `python sync_check.py --audit`.
4. В описании — что изменилось, какие числа сдвинулись и почему.
