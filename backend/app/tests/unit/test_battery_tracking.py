"""Battery tracker history: temperature buckets, C-rate EWMA, charge profile."""

from __future__ import annotations

import pytest

from app.simulation.battery import BatteryTracker


@pytest.fixture
def tracker(bev_battery) -> BatteryTracker:
    return BatteryTracker(vehicle_id="V01", params=bev_battery, soc=0.9, soh=1.0)


def test_temperature_exposure_is_bucketed(tracker):
    tracker.advance(0.5, dt_s=3600.0, temperature_c=40.0, c_rate=0.5)
    tracker.advance(0.5, dt_s=1800.0, temperature_c=5.0, c_rate=0.5)
    exposure = tracker.temperature_seconds
    assert exposure["35_to_45C"] == pytest.approx(3600.0)
    assert exposure["below_10C"] == pytest.approx(1800.0)
    assert sum(exposure.values()) == pytest.approx(5400.0)


def test_c_rate_ewma_smooths_spikes(tracker):
    tracker.advance(1.0, 60.0, 25.0, c_rate=3.0)
    first = tracker.crate_ewma
    assert 0.0 < first < 3.0
    for _ in range(60):
        tracker.advance(1.0, 60.0, 25.0, c_rate=0.2)
    assert tracker.crate_ewma < first
    assert tracker.crate_ewma > 0.0


def test_charge_sessions_are_recorded_and_bounded(tracker):
    for i in range(12):
        tracker.soc = 0.4
        tracker.record_charge_session(
            energy_kwh=10.0 + i, mean_power_kw=50.0, connector="CCS", soc_start=0.4
        )
    assert len(tracker.charge_profile) == 10
    assert tracker.charge_profile[-1]["connector"] == "CCS"
    assert tracker.charge_profile[-1]["soc_end"] == pytest.approx(0.4)


def test_state_export_is_consistent_with_tracker(tracker):
    tracker.advance(7.5, 600.0, 33.0, c_rate=1.2)
    state = tracker.to_state()
    assert state.soc == pytest.approx(tracker.soc)
    assert state.soh == pytest.approx(tracker.soh)
    assert state.equivalent_throughput_kwh == pytest.approx(7.5)
    assert state.usable_energy_kwh == pytest.approx(
        tracker.params.usable_energy_kwh * tracker.soh
    )
    assert state.temperature_c == 33.0
