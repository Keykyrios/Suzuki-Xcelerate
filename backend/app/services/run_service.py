"""Run orchestration: manifests, metrics capture, and evidence export bundles."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app import __version__
from app.config.loader import ConfigBundle, get_config
from app.schemas.decision import Decision
from app.schemas.events import EventEnvelope, RunManifest, SimulatedAction, StateSnapshot
from app.services.state_store import StateStore, get_state_store


def _git_commit(project_root: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def get_run_service(
    cfg: ConfigBundle | None = None, store: StateStore | None = None
) -> "RunService":
    return RunService(cfg or get_config(), store or get_state_store())


class RunService:
    """Turns a recorded state history into run manifests and evidence exports."""

    def __init__(self, cfg: ConfigBundle, store: StateStore):
        self.cfg = cfg
        self.store = store
        self._runs: dict[str, RunManifest] = {}

    # -- manifest --------------------------------------------------------
    def build_manifest(
        self,
        *,
        scenario_id: str,
        seed: int,
        solver_name: str,
        solver_version: str,
        solver_status: str,
        status: str,
        headline_metrics: dict[str, float],
        notes: str = "",
        run_id: str | None = None,
    ) -> RunManifest:
        now = datetime.now(timezone.utc)
        run_id = run_id or f"run-{now.strftime('%Y%m%dT%H%M%SZ')}-{seed}"
        manifest = RunManifest(
            run_id=run_id,
            timestamp_utc=now,
            seed=seed,
            git_commit=_git_commit(self.cfg.root),
            config_hash=self.cfg.hash,
            model_versions={"nexus7": __version__},
            solver_name=solver_name,
            solver_version=solver_version,
            solver_status=solver_status,
            scenario_id=scenario_id,
            status=status,
            headline_metrics=headline_metrics,
            evidence_status="SCENARIO_ASSUMPTION",
            notes=notes,
        )
        self._runs[run_id] = manifest
        return manifest

    def get_manifest(self, run_id: str) -> RunManifest | None:
        return self._runs.get(run_id)

    def list_manifests(self) -> list[RunManifest]:
        return [self._runs[k] for k in sorted(self._runs)]

    # -- evidence bundle -------------------------------------------------
    def export_evidence(
        self,
        out_dir: Path,
        *,
        manifest: RunManifest,
        snapshot: StateSnapshot,
        decision: Decision | None,
        actions: list[SimulatedAction],
        events: list[EventEnvelope],
        parameters: list[dict[str, Any]],
        report: dict[str, Any],
    ) -> Path:
        """Write one JSON bundle plus its SHA-256 sidecar; returns the bundle path."""
        out_dir.mkdir(parents=True, exist_ok=True)
        base = out_dir / manifest.run_id
        bundle = {
            "manifest": manifest.model_dump(mode="json"),
            "snapshot": snapshot.model_dump(mode="json"),
            "decision": decision.model_dump(mode="json") if decision else None,
            "actions": [a.model_dump(mode="json") for a in actions],
            "events": [e.model_dump(mode="json") for e in events],
            "parameters": parameters,
            "report": report,
        }
        path = base.with_suffix(".json")
        blob = json.dumps(bundle, indent=2, sort_keys=True, default=str).encode("utf-8")
        path.write_bytes(blob)

        import hashlib

        digest = hashlib.sha256(blob).hexdigest()
        path.with_suffix(".sha256").write_text(f"{digest}  {path.name}\n", encoding="utf-8")
        return path