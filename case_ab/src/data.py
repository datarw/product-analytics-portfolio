"""Загрузка и проверка Cookie Cats (DataCamp project 184 / Kaggle mirror).

Данные вне git: `data/raw/cookie-cats/cookie_cats.csv`.
Лицензия источника не подтверждена; публикуются только агрегаты и графики,
сам CSV в репозиторий не едет.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from paths import PROCESSED, RAW  # noqa: E402

CSV = RAW / "cookie-cats" / "cookie_cats.csv"
CLEAN_PARQUET = PROCESSED / "cookie_cats.parquet"

COLUMNS = ["userid", "version", "sum_gamerounds", "retention_1", "retention_7"]
VERSIONS = {"gate_30", "gate_40"}


def load_raw() -> pd.DataFrame:
    return pd.read_csv(CSV)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Проверки качества и приведение типов; строки не удаляются без причины."""
    missing = set(COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    df = df[COLUMNS].copy()
    df["version"] = df["version"].astype(str)
    df["retention_1"] = df["retention_1"].astype(bool)
    df["retention_7"] = df["retention_7"].astype(bool)
    df["sum_gamerounds"] = df["sum_gamerounds"].astype("int64")

    assert df["userid"].is_unique, "userid must be unique"
    assert set(df["version"].unique()) == VERSIONS, "unexpected version labels"
    assert df["sum_gamerounds"].ge(0).all(), "negative game rounds"
    assert not df[["retention_1", "retention_7"]].isna().any().any()
    return df.reset_index(drop=True)


def load_clean() -> pd.DataFrame:
    df = clean(load_raw())
    CLEAN_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CLEAN_PARQUET, index=False)
    return df


if __name__ == "__main__":
    df = load_clean()
    print(df.shape)
    print(df.groupby("version").size())
