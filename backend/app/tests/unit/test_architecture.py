"""Architecture boundaries: packages stay separated by their contracts."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

APP_DIR = Path(__file__).resolve().parents[2]
EXTERNAL_FRAMEWORKS = {"fastapi", "starlette", "uvicorn"}


def imports_of(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module.split(".")[0])
    return found


def module_of(path: Path) -> str:
    return ".".join(path.relative_to(APP_DIR).with_suffix("").parts)


def python_modules() -> list[Path]:
    return [p for p in APP_DIR.rglob("*.py") if "tests" not in p.parts]


def test_model_layer_never_imports_the_web_layer():
    offenders = []
    for path in python_modules():
        top = path.relative_to(APP_DIR).parts[0]
        if top in {"simulation", "optimization", "demand", "vision", "economics"}:
            if imports_of(path) & EXTERNAL_FRAMEWORKS:
                offenders.append(str(path.relative_to(APP_DIR)))
    assert not offenders, f"model modules import web frameworks: {offenders}"


def test_data_layer_does_not_depend_on_api_routers():
    for path in APP_DIR.rglob("*.py"):
        if "data_access" in path.parts:
            imports = imports_of(path)
            assert "fastapi" not in imports
            assert not any(name.startswith("routes_") for name in imports)


def test_simulated_actions_never_touch_hardware_sdks():
    hardware_tokens = {"can", "obd", "pyserial", "pymodbus", "raspberry", "gpio"}
    for path in python_modules():
        imports = {name.lower() for name in imports_of(path)}
        assert not (imports & hardware_tokens), f"{path} imports hardware SDK {imports & hardware_tokens}"


def test_every_package_is_importable_from_the_app_root():
    import app  # noqa: F401
    import app.main

    assert app.main.app is not None


@pytest.mark.parametrize(
    "package",
    ["api", "config", "data_access", "demand", "economics", "optimization", "schemas", "services", "simulation", "vision"],
)
def test_subpackages_have_init_files(package):
    assert (APP_DIR / package / "__init__.py").is_file()


def test_no_circular_imports_at_app_startup():
    # Importing the ASMI application pulls in every route, service and model
    # module; a cycle would raise here.
    import importlib

    importlib.import_module("app.main")
    importlib.import_module("app.services.optimization_service")
    importlib.import_module("app.simulation.route_energy")