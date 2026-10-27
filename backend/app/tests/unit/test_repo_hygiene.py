"""Repository hygiene: no task markers, no illustrative payload constants."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]
SHIPPED_TAILS = {".py", ".ts", ".tsx", ".js", ".jsx", ".css", ".html"}
SCAN_DIRS = ("backend", "frontend/src", "experiments", "docs", "config")
SKIP_PARTS = {"node_modules", ".venv", "__pycache__", "dist", "build", ".git"}
MARKER = re.compile(r"\b(TODO|FIXME|XXX|PLACEHOLDER|dummy|hard-?coded?)\b", re.IGNORECASE)
# Values copied from the illustrative decision payload in the specification must
# never be produced by code paths.
ILLUSTRATIVE_VALUES = ("airport_heat_charger_outage_01",)


def shipped_files() -> list[Path]:
    files: list[Path] = []
    for name in SCAN_DIRS:
        base = ROOT / name
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if not path.is_file() or path.suffix not in SHIPPED_TAILS:
                continue
            if SKIP_PARTS & set(path.parts):
                continue
            files.append(path)
    return sorted(files)


def test_every_shipped_source_file_exists_for_the_scan():
    files = shipped_files()
    assert files, "no shipped source files were found to scan"


def test_no_task_markers_in_shipped_code():
    offenders: list[str] = []
    for path in shipped_files():
        if path.name == "test_repo_hygiene.py":
            continue  # this file necessarily contains the markers it looks for
        text = path.read_text(encoding="utf-8", errors="ignore")
        for lineno, line in enumerate(text.splitlines(), start=1):
            if MARKER.search(line):
                offenders.append(f"{path.relative_to(ROOT)}:{lineno}: {line.strip()}")
    assert not offenders, "task markers found:\n" + "\n".join(offenders[:20])


def test_no_illustrative_payload_values_in_non_test_code():
    offenders: list[str] = []
    for path in shipped_files():
        if path.name.startswith("test_"):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for token in ILLUSTRATIVE_VALUES:
            if token in text:
                offenders.append(str(path.relative_to(ROOT)))
    assert not offenders, f"illustrative values found in {offenders}"


def test_no_module_level_todo_docstrings_in_app_package():
    app_dir = ROOT / "backend" / "app"
    for path in app_dir.rglob("*.py"):
        if path.name.startswith("test_"):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        lowered = text.lower()
        for token in ("to be implemented", "not implemented yet", "coming soon", "stub"):
            assert token not in lowered, f"{path.relative_to(ROOT)} contains '{token}'"


def test_backend_package_has_no_empty_modules():
    for path in (ROOT / "backend" / "app").rglob("*.py"):
        if path.name == "__init__.py":
            continue
        assert path.read_text(encoding="utf-8").strip(), f"{path.relative_to(ROOT)} is empty"


@pytest.mark.parametrize("path", ["docker-compose.yml", "requirements.lock", "pyproject.toml"])
def test_build_manifests_exist(path):
    assert (ROOT / path).is_file()


def test_frontend_and_docs_trees_are_scanned():
    assert (ROOT / "docs").exists(), "documentation tree is expected"
    assert (ROOT / "frontend" / "src").exists(), "frontend source tree is expected"