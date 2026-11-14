"""Исполняет ноутбуки с нуля и сохраняет вывод.

Запуск из case_ltv/notebooks или из любого места:
    python ../src/execute_notebooks.py 04-split-calibration-holdout.ipynb ...
"""

import sys
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

NBDIR = Path(__file__).resolve().parents[1] / "notebooks"


def execute(name: str, timeout: int = 1800) -> None:
    path = NBDIR / name
    nb = nbf.read(path, as_version=4)
    client = NotebookClient(
        nb, timeout=timeout, kernel_name="python3",
        resources={"metadata": {"path": str(NBDIR)}},
        allow_errors=False,
    )
    client.execute()
    nbf.write(nb, path)
    print("executed", name)


if __name__ == "__main__":
    names = sys.argv[1:] or [
        "01-load-and-profile.ipynb",
        "02-rfm-table.ipynb",
        "03-cohorts-retention.ipynb",
        "04-split-calibration-holdout.ipynb",
        "05-baselines.ipynb",
        "06-bgnbd-fit.ipynb",
        "07-gamma-gamma.ipynb",
        "08-holdout-validation.ipynb",
        "09-horizon-discounting.ipynb",
        "10-segments-decision.ipynb",
    ]
    for n in names:
        execute(n)
