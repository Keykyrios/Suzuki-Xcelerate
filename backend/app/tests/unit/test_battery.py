"""Battery state, charge acceptance, SOH tracking, and passport output."""

from __future__ import annotations

import pytest

from app.schemas.common import PowertrainType
from app.simulation.battery import (
    BatteryTracker,
    charge_acceptance_kw,
    confidence_label,
    soh_uncertainty,
    wear_per_kwh,
)
from app.simulation.model_config import battery_params


@pytest.fixture
def tracker(bev_battery) -> BatteryTracker:
    return BatteryTracker(vehicle_id="V01", params=bev_battery, soc=0.8, soh=1.0)


def test_charge_acceptance_tapers_with_soc(bev_battery):
    levels = [charge_acceptance_kw(bev_battery, soc) for soc in (0.1, 0.5, 0.8, 0.95, 1.0)]
    assert all(b <= a + 1e-9 for a, b in zip(levels, levels[1:]))
    assert levels[0] == pytest.approx(bev_battery.max_charge_power_dc_kw)
    assert levels[-1] == pytest.approx(
        bev_battery.max_charge_power_dc_kw * bev_battery.taper_min_fraction
    )


def test_ac_acceptance_is_below_dc(bev_battery):
    assert charge_acceptance_kw(bev_battery, 0.5, connector="AC") == pytest.approx(
        bev_battery.max_charge_power_ac_kw * charge_acceptance_kw(bev_battery, 0.5)
        / bev_battery.max_charge_power_dc_kw
    )


def test_advance_moves_soc_and_accumulates_throughput(tracker):
    capacity = tracker.effective_capacity_kwh()
    tracker.advance(energy_delta_kwh=5.5, dt_s=3600.0, temperature_c=30.0, c_rate=0.1)
    expected_soc = 0.8 - 5.5 / capacity
    assert tracker.soc == pytest.approx(expected_soc, abs=1e-9)
    assert tracker.throughput_kwh == pytest.approx(5.5)
    assert tracker.equivalent_cycles == pytest.approx(5.5 / tracker.params.equivalent_cycle_capacity_kwh)


def test_soc_is_clamped_to_valid_bounds(tracker):
    tracker.advance(energy_delta_kwh=1e6, dt_s=60.0, temperature_c=25.0, c_rate=1.0)
    assert 0.0 <= tracker.soc <= 1.0
    tracker.soc = 0.05
    tracker.advance(energy_delta_kwh=-1e6, dt_s=60.0, temperature_c=25.0, c_rate=1.0)
    assert tracker.soc == 1.0


def test_soh_declines_and_never_falls_below_the_floor(tracker):
    start = tracker.soh
    for _ in range(400):
        tracker.advance(4.0, 600.0, temperature_c=45.0, c_rate=2.0)
    assert tracker.soh < start
    assert tracker.soh >= tracker.params.soh_floor


def test_wear_is_monotonic_in_every_stress_axis(bev_battery):
    base = wear_per_kwh(bev_battery, 25.0, 1.0, 0.5, 1.0)
    assert wear_per_kwh(bev_battery, 25.0, 2.0, 0.5, 1.0) > base  # C-rate
    assert wear_per_kwh(bev_battery, 45.0, 1.0, 0.5, 1.0) > base  # temperature
    assert wear_per_kwh(bev_battery, 25.0, 1.0, 0.9, 1.0) > base  # depth of discharge
    assert wear_per_kwh(bev_battery, 25.0, 1.0, 0.5, 0.8) > base  # ageing


def test_wear_does_not_move_below_the_configured_base_for_favourable_conditions(bev_battery):
    gentle = wear_per_kwh(bev_battery, bev_battery.optimal_temp_c, 0.2, 0.2, 1.0)
    assert gentle == pytest.approx(bev_battery.throughput_wear_per_kwh)


def test_uncertainty_grows_with_cycles(bev_battery):
    fresh = soh_uncertainty(bev_battery, 0.0)
    cycled = soh_uncertainty(bev_battery, 2000.0)
    assert cycled > fresh
    assert fresh == pytest.approx(bev_battery.soh_uncertainty_base)


def test_confidence_labels_follow_the_configured_thresholds(bev_battery):
    assert confidence_label(bev_battery, bev_battery.soh_confidence_high_max) == "HIGH"
    assert confidence_label(
        bev_battery,
        max(
            bev_battery.soh_confidence_medium_max,
            bev_battery.soh_confidence_high_max + 1e-6,
        ),
    ) == "MEDIUM"
    assert confidence_label(bev_battery, bev_battery.soh_confidence_medium_max + 0.5) == "LOW"


def test_tracker_reports_battery_state(bev_battery):
    tracker = BatteryTracker(vehicle_id="V01", params=bev_battery, soc=0.7, soh=0.93)
    state = tracker.to_state()
    assert state.soc == pytest.approx(0.7)
    assert state.soh == pytest.approx(0.93)
    assert state.usable_energy_kwh == pytest.approx(bev_battery.usable_energy_kwh * 0.93)
    assert 0.0 <= state.soh_uncertainty <= 1.0
    assert state.max_charge_power_kw > 0.0


def test_charge_sessions_are_recorded_and_capped(tracker):
    for index in range(15):
        tracker.soc = 0.5
        tracker.record_charge_session(20.0, 50.0, "CCS2", 0.4)
        assert len(tracker.charge_profile) <= 10 * (index + 1)
    assert len(tracker.charge_profile) == 10
    assert tracker.charge_profile[-1]["soc_start"] == pytest.approx(0.4)


def test_passport_carries_every_required_field(tracker):
    tracker.record_charge_session(25.0, 60.0, "CCS2", 0.3)
    passport = tracker.passport(replacement_cost_inr_per_kwh=12.0, recent_wear_per_km=1e-5)
    payload = passport.model_dump()
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
    assert passport.evidence_status.value == "MODEL_ESTIMATE"
    assert passport.replacement_horizon_km is None or passport.replacement_horizon_km > 0
    assert passport.marginal_wear_cost_inr_per_kwh > 0.0


def test_replacement_horizon_is_unknown_without_a_wear_rate(tracker):
    assert tracker.replacement_horizon_km(None) is None
    assert tracker.replacement_horizon_km(0.0) is None


def test_temperature_buckets_accrue(tracker):
    tracker.temperature_c = 46.0
    tracker.advance(-5.0, 1800.0, temperature_c=46.0, c_rate=0.3)
    assert tracker.temperature_seconds["above_45C"] == 1800.0
    assert sum(tracker.temperature_seconds.values()) == 1800.0


def test_battery_parameters_come_from_configuration(cfg):
    bev = battery_params(PowertrainType.NEXUS_E, cfg)
    rex = battery_params(PowertrainType.NEXUS_X, cfg)
    assert bev.usable_energy_kwh == pytest.approx(float(cfg.v("battery.bev_usable_energy_kwh")))
    assert rex.usable_energy_kwh == pytest.approx(float(cfg.v("battery.rex_usable_energy_kwh")))
    assert rex.usable_energy_kwh < bev.usable_energy_kwh