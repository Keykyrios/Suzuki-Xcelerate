"""Run manifests: reproducibility records stored with every run.

A manifest pins the seed, configuration hash, model versions, solver
identity and headline metrics so any result can be traced back to exactly the
inputs that produced it.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from app.config.loader import ConfigBundle, find_project_root, get_config
from app.schemas.common import utcnow
from app.schemas.events import RunManifest
from app.simulation.model_config import MODEL_VERSION
from app.simulation.scenario_generator import SCENARIO_VERSION

SOLVER_VERSION = "9.14.6206"


def git_commit(root: Path | None = None) -> str:
    """Current commit hash, or an explicit marker outside a repository."""
    directory = root or find_project_root()
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=directory,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return result.stdout.strip()
    except Exception:  # noqa: BLE001 - missing git is a valid environment
        return "no-git-commit"


def model_versions() -> dict[str, str]:
    versions = {
        "physics": MODEL_VERSION,
        "scenario": SCENARIO_VERSION,
    }
    try:
        import ultralytics

        versions["detector"] = f"ultralytics-{ultralytics.__version__}"
    except Exception:  # noqa: BLE001 - detector stack is optional at import time
        versions["detector"] = "unavailable"
    return versions


class ManifestStore:
    """JSON manifests under data/manifests, one file per run."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or find_project_root()
        self.directory = self.root / "data" / "manifests"
        self.directory.mkdir(parents=True, exist_ok=True)
        self._counter = self._load_counter()

    def _load_counter(self) -> int:
        existing = sorted(self.directory.glob("run_*.json"))
        if not existing:
            return 0
        last = existing[-1].stem.split("_", 1)[-1]
        return int(last) if last.isdigit() else len(existing)

    def next_run_id(self) -> str:
        self._counter += 1
        return f"run_{self._counter:06d}"

    def record(
        self,
        *,
        run_id: str,
        seed: int,
        scenario_id: str,
        solver_name: str,
        solver_status: str,
        headline_metrics: dict[str, float],
        status: str = "completed",
        evidence_status: str = "MODEL_ESTIMATE",
        notes: str = "",
        cfg: ConfigBundle | None = None,
    ) -> RunManifest:
        c = cfg or get_config()
        manifest = RunManifest(
            run_id=run_id,
            timestamp_utc=utcnow(),
            seed=seed,
            git_commit=git_commit(self.root),
            config_hash=c.hash,
            model_versions=model_versions(),
            solver_name=solver_name,
            solver_version=SOLVER_VERSION,
            solver_status=solver_status,
            scenario_id=scenario_id,
            status=status,
            headline_metrics=headline_metrics,
            evidence_status=evidence_status,
            notes=notes,
        )
        path = self.directory / f"{run_id}.json"
        path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        return manifest

    def load(self, run_id: str) -> RunManifest:
        path = self.directory / f"{run_id}.json"
        if not path.is_file():
            raise KeyError(f"unknown run id {run_id}")
        return RunManifest.model_validate_json(path.read_text(encoding="utf-8"))

    def list_runs(self) -> list[str]:
        return sorted(p.stem for p in self.directory.glob("run_*.json"))


def write_benchmark_report(name: str, payload: dict, root: Path | None = None) -> Path:
    """Store a machine-readable benchmark/report artifact."""
    base = root or find_project_root()
    directory = base / "experiments" / "results"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return path
