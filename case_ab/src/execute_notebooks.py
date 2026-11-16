"""Исполняет ноутбуки case_ab с нуля и сохраняет вывод."""

import sys
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

NBDIR = Path(__file__).resolve().parents[1] / "notebooks"


def execute(name: str, timeout: int = 900) -> None:
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
        "11-load-and-aa-check.ipynb",
        "12-effect-and-guardrails.ipynb",
        "13-design-and-power.ipynb",
    ]
    for n in names:
        execute(n)
