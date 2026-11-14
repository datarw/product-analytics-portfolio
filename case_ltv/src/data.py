"""Загрузка и чистка Online Retail II (UCI, CC BY 4.0).

Чистка возвращает не только таблицу, но и журнал шагов: сколько строк и
клиентов удалил каждый шаг. Журнал идёт в README как контрольные точки.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from paths import INTERIM, PROCESSED, RAW  # noqa: E402

XLSX = RAW / "online-retail-ii" / "online_retail_II.xlsx"
RAW_PARQUET = INTERIM / "online_retail_ii_raw.parquet"
CLEAN_PARQUET = PROCESSED / "transactions.parquet"

COLUMNS = {
    "Invoice": "invoice",
    "StockCode": "stock_code",
    "Description": "description",
    "Quantity": "quantity",
    "InvoiceDate": "date",
    "Price": "price",
    "Customer ID": "customer_id",
    "Country": "country",
}

# Служебные коды: почта, комиссии, ручные корректировки, тестовые позиции.
SERVICE_CODES = ("POST", "D", "DOT", "M", "BANK CHARGES", "AMAZONFEE", "CRUK",
                 "S", "B", "TEST001", "TEST002", "ADJUST", "ADJUST2", "PADS", "C2")
SERVICE_PREFIXES = ("gift_",)   # подарочные ваучеры: платёжное средство, не товар


def _is_service(code: pd.Series) -> pd.Series:
    up = code.str.upper()
    return up.isin(SERVICE_CODES) | code.str.lower().str.startswith(SERVICE_PREFIXES)


def load_raw(refresh: bool = False) -> pd.DataFrame:
    """Оба листа xlsx одной таблицей; кэш в interim/ (чтение xlsx ~70 с)."""
    if RAW_PARQUET.exists() and not refresh:
        return pd.read_parquet(RAW_PARQUET)
    sheets = pd.read_excel(XLSX, sheet_name=None, dtype={"Invoice": str, "StockCode": str})
    df = pd.concat(sheets.values(), ignore_index=True).rename(columns=COLUMNS)
    df["stock_code"] = df["stock_code"].astype(str)
    df["description"] = df["description"].astype("string")
    RAW_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(RAW_PARQUET, index=False)
    return df


def _step(log: list, name: str, before: pd.DataFrame, after: pd.DataFrame) -> None:
    log.append({
        "step": name,
        "rows_removed": len(before) - len(after),
        "rows_left": len(after),
        "customers_left": after["customer_id"].nunique(),
    })


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Возвращает (таблица транзакций клиентов, журнал шагов).

    Отмены (счета с префиксом C) не удаляются, а остаются отрицательными
    строками: они нужны для net-выручки. Исключаются только строки, которые
    не могут быть покупкой клиента.
    """
    log = [{"step": "raw", "rows_removed": 0, "rows_left": len(df),
            "customers_left": df["customer_id"].nunique()}]
    df = df.drop_duplicates()
    log.append({"step": "exact duplicates", "rows_removed": log[0]["rows_left"] - len(df),
                "rows_left": len(df), "customers_left": df["customer_id"].nunique()})

    d = df.dropna(subset=["customer_id"])
    _step(log, "null customer_id", df, d)
    df = d

    d = df[df["date"].notna()]
    _step(log, "null date", df, d)
    df = d

    d = df[~_is_service(df["stock_code"])]
    _step(log, "service stock codes", df, d)
    df = d

    d = df[df["price"] > 0]
    _step(log, "price <= 0", df, d)
    df = d

    # Клиент только с отменами: первой покупки в данных нет, t0 не определён.
    has_purchase = df.loc[~df["invoice"].str.startswith("C"), "customer_id"].unique()
    d = df[df["customer_id"].isin(has_purchase)]
    _step(log, "customers with cancellations only", df, d)
    df = d

    df = df.assign(
        customer_id=df["customer_id"].astype("int64"),
        is_cancel=df["invoice"].str.startswith("C"),
    )
    df["revenue"] = df["quantity"] * df["price"]
    return df.reset_index(drop=True), pd.DataFrame(log)


def load_clean(refresh: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    tx, log = clean(load_raw(refresh))
    CLEAN_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    tx.to_parquet(CLEAN_PARQUET, index=False)
    return tx, log


if __name__ == "__main__":
    tx, log = load_clean()
    print(log.to_string(index=False))
