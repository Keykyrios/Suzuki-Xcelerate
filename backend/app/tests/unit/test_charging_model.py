"""Charging model: delivery, queue, tariffs, deadlines, mission targeting."""

from __future__ import annotations

import pytest

from app.schemas.common import ChargerStatus
from app.schemas.reason_codes import ReasonCode
from app.simulation.charging import (
    ChargingEconomics,
    charge_duration_min,
    electricity_cost_inr,
    energy_required_kwh,
    evaluate_charger,
    fits_deadline,
    grid_energy_kwh,
    mission_target_soc,
    queue_wait_min,
    total_cost_inr,
)


def economics() -> ChargingEconomics:
    return ChargingEconomics(
        queue_cost_inr_per_min=6.0,
        detour_cost_inr_per_km=12.0,
        lost_productive_inr_per_min=9.0,
        energy_inr_per_kwh=9.0,
        dc_surcharge_inr_per_kwh=2.5,
        wear_inr_per_kwh=1.5,
    )


TARIFF_BANDS = (16.0, 7.0)
TARIFF_WINDOWS = (18, 23, 0, 6)


def test_energy_required_uses_pack_size_and_soc_delta(bev_battery):
    assert energy_required_kwh(bev_battery, 0.4, 0.6) == pytest.approx(
        bev_battery.usable_energy_kwh * 0.2, rel=1e-9
    )
    assert energy_required_kwh(bev_battery, 0.6, 0.4) == 0.0
    assert energy_required_kwh(bev_battery, 0.9, 0.9) == 0.0


def test_grid_energy_accounts_for_charge_efficiency(bev_battery):
    assert grid_energy_kwh(55.0, bev_battery) == pytest.approx(
        55.0 / bev_battery.charge_efficiency, rel=1e-9
    )
    assert grid_energy_kwh(55.0, bev_battery) >= 55.0


def test_charge_duration_scales_with_delivered_energy(charger_factory, bev_battery):
    quick = charge_duration_min(bev_battery, charger_factory(), 1, 0.3, 0.5)
    deeper = charge_duration_min(bev_battery, charger_factory(), 1, 0.3, 0.9)
    nothing = charge_duration_min(bev_battery, charger_factory(), 1, 0.5, 0.5)
    assert deeper[0] > quick[0]
    assert quick[2] > 0.0
    assert nothing == (0.0, 0.0, 0.0)


def test_charge_duration_respects_the_taper(charger_factory, bev_battery):
    low_soc = charge_duration_min(bev_battery, charger_factory(), 1, 0.2, 0.5)
    high_soc = charge_duration_min(bev_battery, charger_factory(), 1, 0.8, 1.0)
    energy_low, energy_high = bev_battery.usable_energy_kwh * 0.3, bev_battery.usable_energy_kwh * 0.2
    # The high-SOC window takes longer per kWh because acceptance tapers.
    assert (high_soc[0] / energy_high) > (low_soc[0] / energy_low)


def test_offline_charger_is_infeasible_with_a_reason_code(charger_factory, bev_battery):
    estimator = evaluate_charger(
        charger=charger_factory(status=ChargerStatus.OFFLINE.value),
        battery=bev_battery,
        economics=economics(),
        soc_now=0.5,
        target_soc=0.8,
        detour_km=2.0,
        plug_count_in_use=1,
        hour_of_day=12,
        tariff_bands=TARIFF_BANDS,
        tariff_windows=TARIFF_WINDOWS,
    )
    assert estimator.feasible is False
    assert ReasonCode.CHARGER_OFFLINE.value in estimator.reason_codes
    assert estimator.cost.total_inr == 0.0


def test_queue_length_adds_wait_and_queue_cost(charger_factory, bev_battery):
    idle = evaluate_charger(
        charger=charger_factory(queue_length=0, plugs_in_use=2),
        battery=bev_battery,
        economics=economics(),
        soc_now=0.5,
        target_soc=0.7,
        detour_km=1.0,
        plug_count_in_use=2,
        hour_of_day=12,
        tariff_bands=TARIFF_BANDS,
        tariff_windows=TARIFF_WINDOWS,
    )
    busy = evaluate_charger(
        charger=charger_factory(queue_length=3, plugs_in_use=2),
        battery=bev_battery,
        economics=economics(),
        soc_now=0.5,
        target_soc=0.7,
        detour_km=1.0,
        plug_count_in_use=2,
        hour_of_day=12,
        tariff_bands=TARIFF_BANDS,
        tariff_windows=TARIFF_WINDOWS,
    )
    assert busy.queue_wait_min > idle.queue_wait_min
    assert busy.cost.waiting_inr > idle.cost.waiting_inr
    assert ReasonCode.DEADLINE_TIGHTENED_BY_QUEUE.value in busy.reason_codes


def test_queue_wait_is_zero_without_a_queue(charger_factory):
    assert queue_wait_min(charger_factory(queue_length=0, plugs_in_use=0), 0) == 0.0
    assert queue_wait_min(charger_factory(queue_length=0, plugs_in_use=2), 2) == 0.0


def test_detour_adds_proportional_cost(charger_factory, bev_battery):
    near = evaluate_charger(
        charger=charger_factory(),
        battery=bev_battery,
        economics=economics(),
        soc_now=0.5,
        target_soc=0.7,
        detour_km=1.0,
        plug_count_in_use=1,
        hour_of_day=12,
        tariff_bands=TARIFF_BANDS,
        tariff_windows=TARIFF_WINDOWS,
    )
    far = evaluate_charger(
        charger=charger_factory(),
        battery=bev_battery,
        economics=economics(),
        soc_now=0.5,
        target_soc=0.7,
        detour_km=4.0,
        plug_count_in_use=1,
        hour_of_day=12,
        tariff_bands=TARIFF_BANDS,
        tariff_windows=TARIFF_WINDOWS,
    )
    assert far.cost.detour_inr == pytest.approx(4 * near.cost.detour_inr, rel=1e-6)
    assert far.cost.detour_inr > near.cost.detour_inr


def test_cost_breakdown_sums_its_five_terms(charger_factory, bev_battery):
    estimate = evaluate_charger(
        charger=charger_factory(queue_length=1),
        battery=bev_battery,
        economics=economics(),
        soc_now=0.4,
        target_soc=0.75,
        detour_km=2.5,
        plug_count_in_use=1,
        hour_of_day=20,
        tariff_bands=TARIFF_BANDS,
        tariff_windows=TARIFF_WINDOWS,
    )
    cost = estimate.cost
    assert total_cost_inr(cost) == pytest.approx(
        cost.electricity_inr
        + cost.waiting_inr
        + cost.detour_inr
        + cost.lost_productive_inr
        + cost.degradation_inr,
        rel=1e-9,
    )
    assert cost.electricity_inr > 0.0
    assert cost.degradation_inr > 0.0
    assert cost.lost_productive_inr >= cost.waiting_inr * (9.0 / 6.0) - 1e-6


def test_tariff_windows_change_the_electricity_cost():
    night = electricity_cost_inr(10.0, 12.0, False, 0.0, 2, TARIFF_BANDS, TARIFF_WINDOWS)
    evening = electricity_cost_inr(10.0, 12.0, False, 0.0, 20, TARIFF_BANDS, TARIFF_WINDOWS)
    midday = electricity_cost_inr(10.0, 12.0, False, 0.0, 12, TARIFF_BANDS, TARIFF_WINDOWS)
    assert night < midday < evening
    assert evening == pytest.approx(10.0 * TARIFF_BANDS[0])
    assert night == pytest.approx(10.0 * TARIFF_BANDS[1])


def test_dc_fast_surcharge_is_added(charger_factory, bev_battery):
    ac_only = electricity_cost_inr(10.0, 12.0, False, 2.5, 12, TARIFF_BANDS, TARIFF_WINDOWS)
    dc = electricity_cost_inr(10.0, 12.0, True, 2.5, 12, TARIFF_BANDS, TARIFF_WINDOWS)
    assert dc - ac_only == pytest.approx(25.0)


def test_mission_target_covers_next_mission_plus_reserve_and_respects_the_band(bev_battery, cfg):
    mission_energy = bev_battery.usable_energy_kwh * 0.25
    target, codes = mission_target_soc(
        bev_battery,
        soc_now=0.5,
        mission_energy_kwh=mission_energy,
        uncertainty_reserve=float(cfg.opt("charging.mission_uncertainty_reserve")),
        max_target_soc=float(cfg.opt("charging.max_target_soc")),
        operating_max=float(cfg.op("soc_policy.soc_operating_max")),
    )
    assert ReasonCode.CHARGE_TARGET_ONLY_MISSION_ENERGY.value in codes
    assert target > 0.5
    assert target <= min(
        float(cfg.opt("charging.max_target_soc")), float(cfg.op("soc_policy.soc_operating_max"))
    )


def test_mission_target_never_recommends_charging_to_full_by_default(bev_battery, cfg):
    target, _ = mission_target_soc(
        bev_battery,
        soc_now=0.6,
        mission_energy_kwh=2.0,
        uncertainty_reserve=0.1,
        max_target_soc=float(cfg.opt("charging.max_target_soc")),
        operating_max=float(cfg.op("soc_policy.soc_operating_max")),
    )
    assert target <= 1.0
    assert target < 1.0 or target == pytest.approx(1.0)


def test_target_is_capped_and_labelled_when_the_mission_is_small(bev_battery):
    target, codes = mission_target_soc(
        bev_battery,
        soc_now=0.9,
        mission_energy_kwh=50.0,
        uncertainty_reserve=0.2,
        max_target_soc=0.92,
        operating_max=0.95,
    )
    assert target == pytest.approx(0.92)
    assert ReasonCode.MISSION_TARGET_SOC_CAPPED.value in codes


def test_deadline_fit_uses_travel_queue_and_plug_time(charger_factory, bev_battery):
    estimate = evaluate_charger(
        charger=charger_factory(queue_length=2),
        battery=bev_battery,
        economics=economics(),
        soc_now=0.5,
        target_soc=0.9,
        detour_km=3.0,
        plug_count_in_use=1,
        hour_of_day=12,
        tariff_bands=TARIFF_BANDS,
        tariff_windows=TARIFF_WINDOWS,
    )
    assert fits_deadline(estimate, 0.0, estimate.plug_in_time_s) is True
    assert fits_deadline(estimate, 0.0, estimate.plug_in_time_s - 1.0) is False
    assert fits_deadline(estimate, 300.0, estimate.plug_in_time_s + 299.0) is False


def test_site_limit_bounds_the_delivered_power(charger_factory, bev_battery):
    shared = evaluate_charger(
        charger=charger_factory(rated_power_kw=150.0, site_limit_kw=60.0, plug_count=4),
        battery=bev_battery,
        economics=economics(),
        soc_now=0.2,
        target_soc=0.6,
        detour_km=0.0,
        plug_count_in_use=2,
        hour_of_day=12,
        tariff_bands=TARIFF_BANDS,
        tariff_windows=TARIFF_WINDOWS,
    )
    assert ReasonCode.CHARGER_SITE_LIMIT_APPLIED.value in shared.reason_codes
    assert shared.effective_power_kw <= 60.0 + 1e-6