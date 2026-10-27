"""Loads and caches the configuration bundle used by every service."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from app.config.registry import ParameterRegistry, load_registry

CONFIG_FILES = (
    "vehicle",
    "operating_policy",
    "optimizer",
    "economics",
    "demo_scenarios",
)


def find_project_root(start: Path | None = None) -> Path:
    """Locate the project root via the NEXUS7_ROOT env var or the config tree."""
    env_root = os.environ.get("NEXUS7_ROOT")
    if env_root:
        root = Path(env_root).expanduser().resolve()
        if (root / "config" / "vehicle.yaml").is_file():
            return root
    here = (start or Path(__file__)).resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "config" / "vehicle.yaml").is_file():
            return candidate
    raise FileNotFoundError(
        "project root not found: set NEXUS7_ROOT or run from inside the repository"
    )


class ConfigBundle:
    """All configuration registries plus the combined configuration hash."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or find_project_root()
        self.registries: dict[str, ParameterRegistry] = {}
        for name in CONFIG_FILES:
            path = self.root / "config" / f"{name}.yaml"
            if not path.is_file():
                raise FileNotFoundError(f"missing configuration file: {path}")
            self.registries[name] = load_registry(path)
        digest = "".join(r.content_hash() for r in self.registries.values())
        import hashlib

        self.hash = hashlib.sha256(digest.encode("utf-8")).hexdigest()

    # -- typed access -----------------------------------------------------
    @property
    def vehicle(self) -> ParameterRegistry:
        return self.registries["vehicle"]

    @property
    def policy(self) -> ParameterRegistry:
        return self.registries["operating_policy"]

    @property
    def optimizer(self) -> ParameterRegistry:
        return self.registries["optimizer"]

    @property
    def economics(self) -> ParameterRegistry:
        return self.registries["economics"]

    @property
    def demo(self) -> ParameterRegistry:
        return self.registries["demo_scenarios"]

    def v(self, path: str, default: Any = None) -> Any:  # noqa: F821
        return self.vehicle.value(path, default)

    def op(self, path: str, default: Any = None) -> Any:  # noqa: F821
        return self.policy.value(path, default)

    def opt(self, path: str, default: Any = None) -> Any:  # noqa: F821
        return self.optimizer.value(path, default)

    def eco(self, path: str, default: Any = None) -> Any:  # noqa: F821
        return self.economics.value(path, default)

    def demo_v(self, path: str, default: Any = None) -> Any:  # noqa: F821
        return self.demo.value(path, default)

    def snapshot(self) -> list[dict]:
        """Every parameter from every file, for the /api/config endpoint."""
        out: list[dict] = []
        for name, registry in self.registries.items():
            for p in registry.dump():
                p["file"] = name
                out.append(p)
        return out

    def versions(self) -> dict[str, str]:
        return {name: r.version for name, r in self.registries.items()}


@lru_cache(maxsize=1)
def get_config() -> ConfigBundle:
    return ConfigBundle()


def reload_config() -> ConfigBundle:
    """Drop the cached bundle (used by tests that patch the environment)."""
    get_config.cache_clear()
    return get_config()
