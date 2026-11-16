"""Исполняет ноутбуки case_dashboard с нуля и сохраняет вывод."""

import sys
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

NBDIR = Path(__file__).resolve().parents[1] / "notebooks"


def execute(name: str, timeout: int = 600) -> None:
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
    names = sys.argv[1:] or ["14-unit-economics.ipynb"]
    for n in names:
        execute(n)
