"""Versioned parameter registry.

Configuration files are YAML trees whose leaves are parameter records:

    some.path:
      name: human readable name
      value: <any JSON value>
      unit: "kWh"
      status: DESIGN_TARGET
      source_or_rationale: "why this number exists"
      min: 0.0        # optional
      max: 1.0        # optional
      notes: "..."    # optional
      <extra keys>    # optional, carried through to consumers

The registry flattens the tree, validates every leaf, and exposes typed access
so that application code never hard-codes tunable values.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.config.evidence import EvidenceStatus


class ConfigError(ValueError):
    """Raised when a configuration file does not follow the parameter schema."""


class Parameter(BaseModel):
    """One tunable value with provenance metadata."""

    model_config = ConfigDict(extra="allow")

    path: str
    name: str
    value: Any
    unit: str
    status: EvidenceStatus
    source_or_rationale: str
    min: float | None = None
    max: float | None = None
    notes: str | None = None

    @field_validator("unit", "name", "source_or_rationale")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v or not str(v).strip():
            raise ValueError("must be a non-empty string")
        return v

    def check_range(self) -> None:
        if self.min is None or self.max is None:
            return
        if self.min > self.max:
            raise ConfigError(f"{self.path}: min {self.min} > max {self.max}")
        if isinstance(self.value, (int, float)) and not isinstance(self.value, bool):
            if not (self.min <= float(self.value) <= self.max):
                raise ConfigError(
                    f"{self.path}: value {self.value} outside [{self.min}, {self.max}]"
                )


def _is_leaf(node: dict[str, Any]) -> bool:
    return isinstance(node, dict) and "value" in node


class ParameterRegistry:
    """Flattened, validated view of one configuration file."""

    def __init__(self, source_name: str, tree: dict[str, Any]) -> None:
        self.source_name = source_name
        version = tree.get("version")
        if not isinstance(version, str) or not version:
            raise ConfigError(f"{source_name}: missing top-level 'version'")
        self.version = version
        self._parameters: dict[str, Parameter] = {}
        self._sections: dict[str, Any] = {}
        self._walk(tree.get("parameters", tree), prefix="")
        for p in self._parameters.values():
            p.check_range()
        if not self._parameters:
            raise ConfigError(f"{source_name}: no parameters found")

    def _walk(self, node: dict[str, Any], prefix: str) -> None:
        if not isinstance(node, dict):
            raise ConfigError(f"{self.source_name}:{prefix or '<root>'} must be a mapping")
        for key, child in node.items():
            if key == "version" and prefix == "":
                continue
            path = f"{prefix}{key}"
            if _is_leaf(child):
                try:
                    param = Parameter(path=path, **child)
                except Exception as exc:  # noqa: BLE001 - re-raised with file context
                    raise ConfigError(f"{self.source_name}:{path}: {exc}") from exc
                self._parameters[path] = param
            elif isinstance(child, dict):
                self._walk(child, prefix=f"{path}.")
            else:
                raise ConfigError(
                    f"{self.source_name}:{path} is not a parameter record (no 'value' key)"
                )

    # -- access -----------------------------------------------------------
    def __contains__(self, path: str) -> bool:
        return path in self._parameters

    def parameter(self, path: str) -> Parameter:
        try:
            return self._parameters[path]
        except KeyError:
            raise ConfigError(f"{self.source_name}: unknown parameter '{path}'") from None

    def value(self, path: str, default: Any = None) -> Any:
        if path not in self._parameters:
            if default is None:
                raise ConfigError(f"{self.source_name}: unknown parameter '{path}'")
            return default
        return self._parameters[path].value

    def values_with_prefix(self, prefix: str) -> dict[str, Any]:
        return {
            p.path[len(prefix) + 1 :]: p.value
            for p in self._parameters.values()
            if p.path.startswith(f"{prefix}.")
        }

    def all_parameters(self) -> list[Parameter]:
        return [self._parameters[k] for k in sorted(self._parameters)]

    # -- integrity --------------------------------------------------------
    def content_hash(self) -> str:
        payload = {p.path: p.value for p in self.all_parameters()}
        blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()

    def dump(self) -> list[dict[str, Any]]:
        return [p.model_dump(mode="json") for p in self.all_parameters()]


def load_registry(path: Path) -> ParameterRegistry:
    with path.open("r", encoding="utf-8") as fh:
        tree = yaml.safe_load(fh)
    if not isinstance(tree, dict):
        raise ConfigError(f"{path}: root must be a mapping")
    return ParameterRegistry(source_name=path.name, tree=tree)
