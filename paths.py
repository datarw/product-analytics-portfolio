"""Единая точка разрешения путей.

Репозиторий лежит внутри контейнера Product_Analytics_Portfolio/.
Данные и библиотека — соседи репозитория, вне git.

Использование в ноутбуке:
    import sys; sys.path.insert(0, "..")   # из case_*/notebooks/ — ".."/".."
    from paths import RAW, PROCESSED, LIBRARY
    df = pd.read_parquet(PROCESSED / "transactions.parquet")
"""

from pathlib import Path

REPO = Path(__file__).resolve().parent          # product-analytics-portfolio/
CONTAINER = REPO.parent                          # Product_Analytics_Portfolio/

# --- вне git ---
DATA = CONTAINER / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"

LIBRARY = CONTAINER / "ltv-clv-library"
PAPERS = LIBRARY / "papers"
NOTES = LIBRARY / "notes"
DRAFTS = LIBRARY / "drafts"

JOURNALS = CONTAINER / "journals"                # снапшоты case_*.md из веб-проекта

# --- внутри git ---
DOCS = REPO / "docs"
CASE_LTV = REPO / "case_ltv"
CASE_AB = REPO / "case_ab"
CASE_DASHBOARD = REPO / "case_dashboard"

RANDOM_SEED = 42


def figures(case: Path) -> Path:
    """Папка для графиков конкретного кейса. figures(CASE_LTV) -> .../reports/figures"""
    return case / "reports" / "figures"


def ensure_dirs() -> None:
    """Создать недостающие папки вне git. Безопасно вызывать многократно."""
    for d in (RAW, INTERIM, PROCESSED, PAPERS, NOTES, DRAFTS, JOURNALS):
        d.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    ensure_dirs()
    for name in ("REPO", "CONTAINER", "RAW", "PROCESSED", "LIBRARY"):
        print(f"{name:12} {globals()[name]}")
