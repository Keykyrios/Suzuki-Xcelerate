"""Charging economics: cost components, offline rejection, tariff effects."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.schemas.charger import ChargerState
from app.schemas.common import ChargerStatus
from app.schemas.reason_codes import ReasonCode
from app.simulation.charging import (
    ChargingEconomics,
    electricity_cost_inr,
    evaluate_charger,
    total_cost_inr,
)

ECONOMICS = ChargingEconomics(
    queue_cost_inr_per_min=8.0,
    detour_cost_inr_per_km=12.0,
    lost_productive_inr_per_min=3.0,
    energy_inr_per_kwh=9.0,
    dc_surcharge_inr_per_kwh=2.0,
    wear_inr_per_kwh=1.5,
)

TARIFF_WINDOWS = (8, 20, 22, 6)


def make_charger(**overrides) -> ChargerState:
    payload = {
        "charger_id": "CH-01",
        "node_id": "N1",
        "status": ChargerStatus.ONLINE,
        "plug_count": 2,
        "plugs_in_use": 0,
        "rated_power_kw": 60.0,
        "site_limit_kw": 120.0,
        "tariff_inr_per_kwh": 10.0,
        "connectors": ["CCS"],
        "queue_length": 0,
        "queue_estimate_min": 0.0,
        "last_updated_utc": datetime(2026, 1, 1, tzinfo=timezone.utc),
    }
    payload.update(overrides)
    return ChargerState(**payload)


def estimate(charger, **overrides):
    kwargs = {
        "charger": charger,
        "battery": None,
        "economics": ECONOMICS,
        "soc_now": 0.5,
        "target_soc": 0.8,
        "detour_km": 3.0,
        "plug_count_in_use": 0,
        "hour_of_day": 12,
        "tariff_bands": (12.0, 8.0),
        "tariff_windows": TARIFF_WINDOWS,
    }
    kwargs.update(overrides)
    return evaluate_charger(**kwargs)


def test_total_is_the_sum_of_all_components(bev_battery):
    est = estimate(make_charger(), battery=bev_battery)
    cost = est.cost
    expected = (
        cost.electricity_inr
        + cost.waiting_inr
        + cost.detour_inr
        + cost.lost_productive_inr
        + cost.degradation_inr
    )
    assert total_cost_inr(cost) == pytest.approx(expected)
    assert cost.electricity_inr > 0.0
    assert cost.degradation_inr > 0.0


def test_offline_charger_is_infeasible_with_a_reason_code(bev_battery):
    est = estimate(make_charger(status=ChargerStatus.OFFLINE), battery=bev_battery)
    assert est.feasible is False
    assert ReasonCode.CHARGER_OFFLINE.value in est.reason_codes


def test_queue_adds_wait_and_cost(bev_battery):
    idle = estimate(make_charger(), battery=bev_battery)
    busy = estimate(
        make_charger(queue_length=4, queue_estimate_min=6.0), battery=bev_battery
    )
    assert busy.queue_wait_min > idle.queue_wait_min
    assert busy.cost.waiting_inr > idle.cost.waiting_inr
    assert ReasonCode.DEADLINE_TIGHTENED_BY_QUEUE.value in busy.reason_codes


def test_peak_hour_tariff_costs_more_than_off_peak(bev_battery):
    peak = electricity_cost_inr(50.0, 10.0, is_dc_fast=True, dc_surcharge_inr_per_kwh=2.0, hour_of_day=12, bands=(14.0, 6.0), windows=TARIFF_WINDOWS)
    off_peak = electricity_cost_inr(50.0, 10.0, is_dc_fast=True, dc_surcharge_inr_per_kwh=2.0, hour_of_day=2, bands=(14.0, 6.0), windows=TARIFF_WINDOWS)
    assert peak > off_peak


def test_duration_and_power_are_positive_for_a_sane_session(bev_battery):
    est = estimate(make_charger(), battery=bev_battery)
    assert est.feasible is True
    assert est.charge_duration_min > 0.0
    assert est.effective_power_kw > 0.0
    assert est.energy_kwh == pytest.approx((0.8 - 0.5) * bev_battery.usable_energy_kwh, rel=1e-3)
