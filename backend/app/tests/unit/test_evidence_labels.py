"""Evidence taxonomy: eight labels used consistently across config and outputs."""

from __future__ import annotations

import pytest

from app.config.evidence import MODEL_OUTPUT_LABELS, EvidenceStatus
from app.config.loader import get_config
from app.schemas.common import EvidenceLabel


@pytest.fixture(scope="module")
def cfg():
    return get_config()


def test_taxonomy_has_eight_labels():
    labels = {status.value for status in EvidenceStatus}
    assert len(labels) == 8
    assert labels == {
        "OBSERVED_PUBLIC",
        "MEASURED_LOCAL",
        "SYNTHETIC",
        "SCENARIO_ASSUMPTION",
        "DESIGN_TARGET",
        "MODEL_ESTIMATE",
        "RECONSTRUCTED",
        "UNCALIBRATED",
    }


def test_payload_labels_mirror_config_labels():
    assert {e.value for e in EvidenceLabel} == {e.value for e in EvidenceStatus}


def test_every_config_parameter_uses_a_taxonomy_label(cfg):
    for registry in cfg.registries.values():
        for param in registry.all_parameters():
            assert param.status in EvidenceStatus


def test_model_output_labels_are_marked_as_such():
    assert MODEL_OUTPUT_LABELS == {EvidenceStatus.MODEL_ESTIMATE, EvidenceStatus.UNCALIBRATED}


def test_operating_policy_declares_model_status_and_calibration_flag(cfg):
    assert cfg.op("evidence.default_model_status") in {s.value for s in EvidenceStatus}
    assert cfg.op("evidence.calibration_flag") == EvidenceStatus.UNCALIBRATED.value
    phrase = cfg.op("evidence.display_phrase")
    assert "simulation estimate" in phrase
    assert "not vehicle-calibrated" in phrase


def test_synthetic_fixture_is_labelled_synthetic(cfg):
    provenance = cfg.demo.parameter("world.provenance") if "world.provenance" in cfg.demo else None
    if provenance is not None:
        assert provenance.value in {s.value for s in EvidenceStatus}


def test_route_result_defaults_to_uncalibrated_model_estimate():
    from app.schemas.vehicle import RouteResult

    field = RouteResult.model_fields["evidence_status"]
    assert field.default == EvidenceLabel.MODEL_ESTIMATE
    assert RouteResult.model_fields["calibrated"].default is False


def test_battery_passport_declares_estimate_status_and_disclaimer():
    from app.simulation.battery import BatteryTracker
    from app.simulation.model_config import battery_params
    from app.schemas.common import PowertrainType

    tracker = BatteryTracker(
        vehicle_id="V01",
        params=battery_params(PowertrainType.NEXUS_E),
        soc=0.8,
        soh=0.95,
    )
    passport = passport_for(tracker)
    assert passport.evidence_status == EvidenceLabel.MODEL_ESTIMATE
    assert "not a certified diagnosis" in passport.notes
    assert passport.confidence.value in {"HIGH", "MEDIUM", "LOW"}


def passport_for(tracker):
    return tracker.passport(replacement_cost_inr_per_kwh=10.0)