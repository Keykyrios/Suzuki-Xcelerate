"""Run manifests and evidence bundles: reproducibility records."""

from __future__ import annotations

import hashlib
import json

import pytest

from app import __version__
from app.data_access.manifests import ManifestStore, write_benchmark_report
from app.schemas.events import RunManifest
from app.services.run_service import RunService
from app.services.state_store import StateStore


@pytest.fixture
def store(tmp_path) -> ManifestStore:
    return ManifestStore(tmp_path)


def test_run_ids_are_sequential_and_formatted(store):
    first = store.next_run_id()
    second = store.next_run_id()
    assert first == "run_000001"
    assert second == "run_000002"


def test_manifest_records_every_reproducibility_field(store, cfg):
    manifest = store.record(
        run_id="run_000042",
        seed=7,
        scenario_id="demo_scenario",
        solver_name="CP-SAT",
        solver_status="FEASIBLE",
        headline_metrics={"contribution_inr": 1234.5},
    )
    assert isinstance(manifest, RunManifest)
    assert manifest.seed == 7
    assert manifest.config_hash == cfg.hash
    assert manifest.git_commit
    assert manifest.solver_version
    assert manifest.model_versions["physics"]
    assert manifest.model_versions["scenario"]
    assert manifest.headline_metrics["contribution_inr"] == pytest.approx(1234.5)
    assert manifest.evidence_status == "MODEL_ESTIMATE"
    assert manifest.scenario_id == "demo_scenario"


def test_manifest_file_round_trips(store):
    store.record(
        run_id="run_000007",
        seed=1,
        scenario_id="demo",
        solver_name="CP-SAT",
        solver_status="OPTIMAL",
        headline_metrics={"loss_inr": -10.0},
    )
    loaded = store.load("run_000007")
    assert loaded.run_id == "run_000007"
    assert loaded.headline_metrics["loss_inr"] == pytest.approx(-10.0)
    assert store.list_runs() == ["run_000007"]


def test_unknown_run_is_an_explicit_key_error(store):
    with pytest.raises(KeyError):
        store.load("run_999999")
    assert store.list_runs() == []


def test_counter_survives_existing_files(store):
    store.record(
        run_id="run_000003",
        seed=1,
        scenario_id="demo",
        solver_name="CP-SAT",
        solver_status="FEASIBLE",
        headline_metrics={},
    )
    shorter = ManifestStore(store.root)
    assert shorter.list_runs() == ["run_000003"]


def test_benchmark_report_is_written_as_json(tmp_path):
    path = write_benchmark_report(
        "unit_benchmark", {"seed": 1, "metrics": {"a": 1.5}}, root=tmp_path
    )
    assert path.parent == tmp_path / "experiments" / "results"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["seed"] == 1
    assert payload["metrics"]["a"] == pytest.approx(1.5)


def test_run_service_builds_a_manifest(cfg):
    service = RunService(cfg, StateStore())
    manifest = service.build_manifest(
        scenario_id="demo",
        seed=42,
        solver_name="CP-SAT",
        solver_version="9.14",
        solver_status="OPTIMAL",
        status="completed",
        headline_metrics={"contribution_inr": 5.0},
        notes="unit run",
    )
    assert manifest.config_hash == cfg.hash
    assert manifest.git_commit
    assert manifest.model_versions["nexus7"] == __version__
    assert service.get_manifest(manifest.run_id) == manifest
    listed = service.list_manifests()
    assert [m.run_id for m in listed] == [manifest.run_id]


def test_evidence_bundle_is_written_and_hashed(cfg, tmp_path):
    from app.schemas.events import SimulatedActionType

    service = RunService(cfg, StateStore())
    manifest = service.build_manifest(
        scenario_id="demo",
        seed=1,
        solver_name="CP-SAT",
        solver_version="9.14",
        solver_status="FEASIBLE",
        status="completed",
        headline_metrics={},
    )
    snapshot = service.store.snapshot()
    path = service.export_evidence(
        out_dir=tmp_path / "evidence",
        manifest=manifest,
        snapshot=snapshot,
        decision=None,
        actions=[],
        events=service.store.recent_events(),
        parameters=[{"path": "a.b", "value": 1.0}],
        report={"checks": {"unit": "pass"}},
    )
    assert path.is_file()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert set(payload) == {
        "manifest",
        "snapshot",
        "decision",
        "actions",
        "events",
        "parameters",
        "report",
    }
    assert payload["manifest"]["run_id"] == manifest.run_id
    sidecar = path.with_suffix(".sha256")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert sidecar.read_text(encoding="utf-8").split()[0] == digest
    assert SimulatedActionType.CABIN_MODE_CHANGE is not None