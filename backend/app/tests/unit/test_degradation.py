"""Degeneration proxy: marginal wear cost used by dispatch and charging."""

from __future__ import annotations

import pytest

from app.schemas.common import PowertrainType
from app.simulation.battery import BatteryTracker, wear_per_kwh
from app.simulation.model_config import battery_params


@pytest.fixture(scope="module")
def params(cfg):
    return battery_params(PowertrainType.NEXUS_E, cfg)


def test_marginal_wear_cost_is_monotonic_in_throughput_stress(params, cfg):
    replacement = float(cfg.eco("capital.battery_replacement_inr_per_kwh"))
    tracker = BatteryTracker(vehicle_id="V01", params=params, soc=0.8, soh=1.0)
    cool = tracker.marginal_wear_cost_inr_per_kwh(replacement)
    tracker.temperature_c = 50.0
    tracker.crate_ewma = 3.0
    hot = tracker.marginal_wear_cost_inr_per_kwh(replacement)
    assert hot > cool > 0.0


def test_low_soh_raises_the_wear_cost(params, cfg):
    replacement = float(cfg.eco("capital.battery_replacement_inr_per_kwh"))
    healthy = BatteryTracker(vehicle_id="V01", params=params, soc=0.8, soh=1.0)
    aged = BatteryTracker(vehicle_id="V02", params=params, soc=0.8, soh=0.75)
    assert aged.marginal_wear_cost_inr_per_kwh(replacement) > healthy.marginal_wear_cost_inr_per_kwh(
        replacement
    )


def test_wear_sign_is_never_negative(params):
    for temperature in (-10.0, 25.0, 60.0):
        for c_rate in (0.0, 1.0, 5.0):
            for dod in (0.0, 0.5, 1.0):
                assert wear_per_kwh(params, temperature, c_rate, dod, 1.0) >= 0.0


def test_damage_index_accumulates_only_with_throughput(params):
    tracker = BatteryTracker(vehicle_id="V01", params=params, soc=0.8, soh=1.0)
    start = tracker.damage_index
    tracker.advance(0.0, 60.0, temperature_c=30.0, c_rate=1.0)
    assert tracker.damage_index == pytest.approx(start)
    tracker.advance(-2.0, 60.0, temperature_c=30.0, c_rate=1.0)
    assert tracker.damage_index > start


def test_degradation_coefficients_are_configuration_driven(params, cfg):
    assert params.throughput_wear_per_kwh == pytest.approx(
        float(cfg.v("degradation.throughput_wear_per_kwh"))
    )
    assert params.wear_acceleration_factor == pytest.approx(
        float(cfg.v("degradation.wear_acceleration_factor"))
    )
    assert params.equivalent_cycle_capacity_kwh == pytest.approx(
        float(cfg.v("degradation.equivalent_cycle_capacity_kwh"))
    )