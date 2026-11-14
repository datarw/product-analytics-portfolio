#!/usr/bin/env python3
"""sync_check.py — что устарело в knowledge веб-проектов и что сломано в репо.

Кладётся в корень репозитория product-analytics-portfolio/.
Состояние хранится в <контейнер>/journals/sync_state.json (вне git).

Использование:
    python sync_check.py               отчёт: что залить заново
    python sync_check.py --brief       одна строка (для хука и планировщика)
    python sync_check.py --audit       + гигиена репозитория
    python sync_check.py --json        машинный вывод
    python sync_check.py --mark        отметить всё как залитое
    python sync_check.py --mark conventions.md datasets.md
    python sync_check.py --install-hook   поставить git pre-push
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent
CONTAINER = REPO.parent
STATE = CONTAINER / "journals" / "sync_state.json"

# файл -> в каком веб-проекте лежит его копия
TRACKED = {
    "conventions.md": "Product Analytics Portfolio",
    "datasets.md": "Product Analytics Portfolio",
    "case_ltv.md": "Product Analytics Portfolio",
    "case_ab.md": "Product Analytics Portfolio",
    "case_dashboard.md": "Product Analytics Portfolio",
    "cv_claims_map.md": "Search for Work",
}

# журналы заливаем не при каждой правке, а когда в них появился смысл
JOURNALS = {"case_ltv.md", "case_ab.md", "case_dashboard.md"}
SIGNAL_HEADINGS = ("## Журнал решений", "## Что не сработало")

SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__",
             ".ipynb_checkpoints", "raw", "interim", "processed", "papers"}

CASES = ("case_ltv", "case_ab", "case_dashboard")
REQUIRED_SECTIONS = ["## Вопрос", "## Данные", "## Определение метрики",
                     "## Метод", "## Результаты",
                     "## Что не сработало", "## Рекомендация", "## Ограничения",
                     "## Воспроизведение"]


# ---------------------------------------------------------------- утилиты

def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def find_tracked() -> dict[str, Path]:
    """Ищем отслеживаемые файлы по имени во всём контейнере."""
    found: dict[str, Path] = {}
    for p in CONTAINER.rglob("*.md"):
        if p.name in TRACKED and not (set(p.parts) & SKIP_DIRS):
            found.setdefault(p.name, p)
    return found


def load_state() -> dict:
    if STATE.exists():
        try:
            return json.loads(STATE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


def save_state(state: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def journal_has_signal(p: Path) -> bool:
    """В журнале есть содержание: датированная запись или буллет, а не только шаблон."""
    text = p.read_text(encoding="utf-8", errors="ignore")
    entry = re.compile(r"^\s*(?:\d{4}-\d{2}-\d{2}|[-*]\s+\S)")
    for h in SIGNAL_HEADINGS:
        i = text.find(h)
        if i == -1:
            continue
        block = text[i + len(h):].split("\n## ")[0].split("\n---")[0]
        in_fence = False
        for ln in block.splitlines():
            if ln.strip().startswith("```"):
                in_fence = not in_fence
                continue
            if in_fence or "⬜" in ln:
                continue
            if entry.match(ln):
                return True
    return False


# ---------------------------------------------------------------- проверка

def check() -> list[dict]:
    state, found, out = load_state(), find_tracked(), []
    for name, project in TRACKED.items():
        p = found.get(name)
        if p is None:
            out.append({"file": name, "project": project, "status": "missing",
                        "why": "файл не найден локально"})
            continue
        cur, rec = sha(p), state.get(name, {})
        if not rec:
            if name in JOURNALS and not journal_has_signal(p):
                continue  # пустой журнал заливать незачем
            out.append({"file": name, "project": project, "status": "never",
                        "why": "ни разу не заливался"})
        elif rec.get("sha") != cur:
            if name in JOURNALS and not journal_has_signal(p):
                continue
            out.append({"file": name, "project": project, "status": "stale",
                        "why": f"изменён после заливки {rec.get('uploaded', '?')[:10]}"})
    return out


def audit() -> list[str]:
    problems = []

    if (REPO / "data").exists():
        problems.append("data/ лежит внутри репозитория — должна быть на уровне контейнера")
    if not (CONTAINER / "data").exists():
        problems.append("нет <контейнер>/data — запусти python paths.py")

    gi = REPO / ".gitignore"
    if not gi.exists():
        problems.append("нет .gitignore")
    else:
        gi_text = gi.read_text(encoding="utf-8", errors="ignore")
        for need in ("data/", "__pycache__", ".venv", ".ipynb_checkpoints"):
            if need not in gi_text:
                problems.append(f".gitignore без {need}")

    for case in CASES:
        d = REPO / case
        if not d.exists():
            problems.append(f"нет папки {case}/")
            continue
        rd = d / "README.md"
        if not rd.exists():
            problems.append(f"{case}/README.md отсутствует")
            continue
        text = rd.read_text(encoding="utf-8", errors="ignore")
        missing = [s for s in REQUIRED_SECTIONS if s not in text]
        if missing:
            problems.append(f"{case}/README.md без секций: {', '.join(missing)}")
        if "**TL;DR" not in text:
            problems.append(f"{case}/README.md без TL;DR")
        for rel in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text):
            if rel.startswith("http"):
                continue
            if not (d / rel).exists():
                problems.append(f"{case}/README.md ссылается на несуществующий {rel}")

    # литеральные пути к данным вместо paths.py
    for p in REPO.rglob("*.py"):
        if set(p.parts) & SKIP_DIRS or p.name == "sync_check.py":
            continue
        if re.search(r'["\'][^"\']*\.\./\.\./data/', p.read_text(encoding="utf-8", errors="ignore")):
            problems.append(f"{p.relative_to(REPO)}: литеральный путь к data вместо paths.py")

    req = REPO / "requirements.txt"
    if req.exists():
        loose = [ln.strip() for ln in req.read_text(encoding="utf-8").splitlines()
                 if ln.strip() and not ln.startswith("#") and "==" not in ln]
        if loose:
            problems.append(f"requirements.txt без версий: {', '.join(loose[:5])}")

    return problems


# ---------------------------------------------------------------- вывод

def mark(names: list[str]) -> None:
    state, found = load_state(), find_tracked()
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    targets = names or [s["file"] for s in check() if s["status"] != "missing"]
    marked = []
    for n in targets:
        p = found.get(n)
        if p:
            state[n] = {"sha": sha(p), "uploaded": now, "path": str(p.relative_to(CONTAINER))}
            marked.append(n)
    save_state(state)
    print("отмечено залитым: " + (", ".join(marked) if marked else "ничего не найдено"))


def main() -> int:
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--brief", action="store_true")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--audit", action="store_true")
    ap.add_argument("--mark", nargs="*", default=None)
    ap.add_argument("--install-hook", action="store_true")
    a = ap.parse_args()

    if a.install_hook:
        hook = REPO / ".git" / "hooks" / "pre-push"
        if not hook.parent.exists():
            print("нет .git/hooks — запусти git init"); return 1
        hook.write_text('#!/bin/sh\npython "$(git rev-parse --show-toplevel)/sync_check.py" '
                        '--brief --audit\nexit 0\n', encoding="utf-8")
        hook.chmod(0o755)
        print(f"хук поставлен: {hook}")
        return 0

    if a.mark is not None:
        mark(a.mark); return 0

    stale, problems = check(), (audit() if a.audit else [])

    if a.json:
        print(json.dumps({"stale": stale, "problems": problems}, ensure_ascii=False, indent=2))
        return 0

    if a.brief:
        bits = []
        if stale:
            bits.append("залить заново: " + ", ".join(
                f"{s['file']}→{s['project']}" for s in stale))
        if problems:
            bits.append(f"репо: {len(problems)} замечан.")
        print(" | ".join(bits) if bits else "синхронизировано, замечаний нет")
        return 0

    if stale:
        print("⟳ Обновить в knowledge:")
        for s in stale:
            print(f"  {s['file']:22} → {s['project']:28} {s['why']}")
        print("\nпосле заливки: python sync_check.py --mark")
    else:
        print("knowledge синхронизирован")

    if a.audit:
        print()
        if problems:
            print("Замечания по репозиторию:")
            for x in problems:
                print(f"  · {x}")
        else:
            print("Репозиторий: замечаний нет")

    return 0


if __name__ == "__main__":
    sys.exit(main())
