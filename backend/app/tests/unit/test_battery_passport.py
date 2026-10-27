"""Battery passport: required fields, uncertainty, wear cost, honesty labels."""

from __future__ import annotations

import pytest

from app.schemas.common import ConfidenceLabel, EvidenceLabel
from app.simulation.battery import BatteryTracker


@pytest.fixture
def tracker(bev_battery) -> BatteryTracker:
    t = BatteryTracker(vehicle_id="V01", params=bev_battery, soc=0.8, soh=0.95)
    t.record_charge_session(energy_kwh=12.0, mean_power_kw=40.0, connector="CCS", soc_start=0.4)
    t.advance(9.0, 900.0, 40.0, c_rate=1.5)
    return t


def test_passport_carries_every_required_field(tracker):
    passport = tracker.passport(replacement_cost_inr_per_kwh=4500.0)
    payload = passport.model_dump(mode="json")
    for field in (
        "vehicle_id",
        "soh",
        "soh_uncertainty",
        "confidence",
        "equivalent_throughput_kwh",
        "equivalent_cycles",
        "temperature_exposure",
        "recent_charge_profile",
        "marginal_wear_cost_inr_per_kwh",
        "replacement_horizon_km",
        "evidence_status",
        "notes",
    ):
        assert field in payload
    assert passport.vehicle_id == "V01"
    assert passport.recent_charge_profile[0]["connector"] == "CCS"


def test_passport_labels_itself_honestly(tracker):
    passport = tracker.passport(replacement_cost_inr_per_kwh=4500.0)
    assert passport.evidence_status is EvidenceLabel.MODEL_ESTIMATE
    assert isinstance(passport.confidence, ConfidenceLabel)
    assert "not a certified diagnosis" in passport.notes


def test_wear_cost_is_positive_and_horizon_requires_a_wear_rate(tracker):
    passport = tracker.passport(replacement_cost_inr_per_kwh=4500.0, recent_wear_per_km=None)
    assert passport.marginal_wear_cost_inr_per_kwh > 0.0
    assert passport.replacement_horizon_km is None
    with_rate = tracker.passport(replacement_cost_inr_per_kwh=4500.0, recent_wear_per_km=0.0005)
    assert with_rate.replacement_horizon_km is not None
    assert with_rate.replacement_horizon_km > 0.0


def test_more_cycles_do_not_reduce_uncertainty(tracker):
    early = tracker.current_uncertainty()
    for _ in range(50):
        tracker.advance(4.0, 600.0, 35.0, c_rate=1.0)
    assert tracker.current_uncertainty() >= early
