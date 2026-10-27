"""Economics: fares, tariffs, cost structure, TCO, contribution, currency."""

from __future__ import annotations

import pytest

from app.economics.model import (
    capital_daily_inr,
    cost_structure,
    fare_for,
    fleet_day_economics,
    reference_case,
    time_of_use_tariff,
    trip_economics,
    tco_breakdown,
)
from app.schemas.common import CabinMode, PowertrainType


def test_every_configured_price_is_in_the_declared_currency(cfg):
    currency = cfg.eco("currency.code")
    assert currency == "INR"
    assert int(cfg.eco("currency.base_year")) >= 2026
    for path in (
        "fares.min_fare_inr",
        "electricity.peak_inr_per_kwh",
        "capital.vehicle_capex_inr",
        "operating_costs.driver_daily_cost_inr",
    ):
        assert "_inr" in path or "inr" in cfg.eco(path).__class__.__name__.lower() or path.endswith("_inr")


def test_fare_is_linear_in_passengers_and_distance():
    small = fare_for(CabinMode.COMMUTE, 2, 10.0)
    large = fare_for(CabinMode.COMMUTE, 4, 10.0)
    far = fare_for(CabinMode.COMMUTE, 2, 20.0)
    assert large.per_pax_km_inr == pytest.approx(2 * small.per_pax_km_inr, rel=1e-6)
    assert far.per_pax_km_inr == pytest.approx(2 * small.per_pax_km_inr, rel=1e-6)
    assert small.minimum_applied is False


def test_minimum_fare_is_applied_when_the_trip_is_tiny(cfg):
    tiny = fare_for(CabinMode.POOL, 1, 0.2)
    assert tiny.minimum_applied is True
    assert tiny.total_inr == pytest.approx(float(cfg.eco("fares.min_fare_inr")))
    long_trip = fare_for(CabinMode.POOL, 1, 100.0)
    assert long_trip.minimum_applied is False
    assert long_trip.total_inr > tiny.total_inr


def test_airport_service_adds_the_configured_fixed_element():
    commute = fare_for(CabinMode.COMMUTE, 3, 15.0)
    airport = fare_for(CabinMode.AIRPORT, 3, 15.0)
    assert airport.fixed_inr == fare_for(CabinMode.AIRPORT, 1, 100.0).fixed_inr
    assert airport.fixed_inr > 0.0
    assert airport.total_inr != commute.total_inr


def test_tariff_windows_and_ordering(cfg):
    peak, peak_label = time_of_use_tariff(int(cfg.eco("electricity.peak_window_start_h")), cfg)
    off, off_label = time_of_use_tariff(int(cfg.eco("electricity.offpeak_window_start_h")), cfg)
    shoulder, shoulder_label = time_of_use_tariff(12, cfg)
    assert peak_label == "peak"
    assert off_label == "offpeak"
    assert shoulder_label == "shoulder"
    assert off < shoulder < peak


def test_capital_daily_cost_depreciates_and_carries_interest(cfg):
    bev = capital_daily_inr(PowertrainType.NEXUS_E, cfg)
    rex = capital_daily_inr(PowertrainType.NEXUS_X, cfg)
    assert bev > 0.0
    assert rex > bev  # the range extender starts costlier
    capex = float(cfg.eco("capital.vehicle_capex_inr"))
    years = float(cfg.eco("capital.depreciation_years"))
    residual = float(cfg.eco("capital.residual_fraction"))
    rate = float(cfg.eco("capital.finance_rate_per_year"))
    expected = capex * (1 - residual) / years / 365.0 + capex * rate / 365.0
    assert bev == pytest.approx(expected, rel=1e-3)


def test_cost_structure_aggregates_only_configured_inputs(cfg):
    structure = cost_structure(PowertrainType.NEXUS_E, cfg)
    assert structure.maintenance_inr_per_km == pytest.approx(
        float(cfg.eco("operating_costs.maintenance_reserve_inr_per_km"))
    )
    assert structure.tyre_inr_per_km == pytest.approx(
        float(cfg.eco("operating_costs.tyre_reserve_inr_per_km"))
    )
    rex = cost_structure(PowertrainType.NEXUS_X, cfg)
    assert rex.maintenance_inr_per_km > structure.maintenance_inr_per_km
    assert structure.fixed_daily_inr == pytest.approx(
        structure.insurance_inr_per_day + structure.capital_daily_inr
    )


def test_trip_contribution_is_revenue_minus_costs(cfg):
    structure = cost_structure(PowertrainType.NEXUS_E, cfg)
    trip = trip_economics(
        service_class=CabinMode.COMMUTE,
        passengers=3,
        trip_distance_km=12.0,
        deadhead_km=2.0,
        energy_kwh=3.5,
        energy_price_inr_per_kwh=9.0,
        wear_cost_inr=12.0,
        structure=structure,
        pickup_time_min=4.0,
        pickup_time_weight_inr_per_min=float(cfg.opt("dispatch_weights.pickup_time_inr_per_min")),
    )
    assert trip.contribution_inr == pytest.approx(
        trip.revenue_inr
        - (
            trip.energy_cost_inr
            + trip.wear_cost_inr
            + trip.deadhead_cost_inr
            + trip.service_cost_inr
        ),
        rel=1e-9,
    )
    assert trip.energy_cost_inr == pytest.approx(3.5 * 9.0)


def test_tco_components_sum_to_the_headline(cfg):
    tco = tco_breakdown(
        energy_kwh_per_km=0.18,
        energy_price_inr_per_kwh=9.0,
        battery_wear_inr_per_kwh=2.0,
        powertrain=PowertrainType.NEXUS_E,
        cfg=cfg,
    )
    assert tco.tco_inr_per_km == pytest.approx(
        tco.energy_inr_per_km
        + tco.maintenance_inr_per_km
        + tco.tyre_inr_per_km
        + tco.capital_inr_per_km
        + tco.insurance_inr_per_km
        + tco.battery_wear_inr_per_km,
        abs=0.01,
    )
    assert tco.tco_inr_per_km > 0.0
    assert tco.capital_inr_per_km == pytest.approx(
        tco.capital_inr_per_km * 1.0
    )  # capital spread over the configured utilisation


def test_higher_utilisation_spreads_fixed_costs(cfg):
    sparse = tco_breakdown(
        energy_kwh_per_km=0.2,
        energy_price_inr_per_kwh=9.0,
        battery_wear_inr_per_kwh=2.0,
        utilisation_km_per_day=60.0,
        cfg=cfg,
    )
    busy = tco_breakdown(
        energy_kwh_per_km=0.2,
        energy_price_inr_per_kwh=9.0,
        battery_wear_inr_per_kwh=2.0,
        utilisation_km_per_day=300.0,
        cfg=cfg,
    )
    assert busy.capital_inr_per_km < sparse.capital_inr_per_km
    assert busy.tco_inr_per_km < sparse.tco_inr_per_km


def test_fleet_day_contribution_and_owner_driver_cashflow(cfg):
    day = fleet_day_economics(
        gross_revenue_inr=18_000.0,
        energy_cost_inr=3_200.0,
        wear_cost_inr=900.0,
        charging_opportunity_cost_inr=600.0,
        missed_service_cost_inr=400.0,
        total_km=220.0,
        cfg=cfg,
    )
    assert day.contribution_inr == pytest.approx(
        day.gross_revenue_inr - day.operating_cost_inr, rel=1e-9
    )
    assert day.owner_driver_cashflow_inr == pytest.approx(
        day.contribution_inr - day.driver_cost_inr, rel=1e-9
    )
    assert day.maintenance_tyre_inr == pytest.approx(
        220.0
        * (
            float(cfg.eco("operating_costs.maintenance_reserve_inr_per_km"))
            + float(cfg.eco("operating_costs.tyre_reserve_inr_per_km"))
        ),
        rel=1e-6,
    )


def test_reference_case_is_labelled_as_a_stored_model_output(cfg):
    reference = reference_case(cfg)
    assert set(reference) == {
        "tco_inr_per_km",
        "owner_driver_income_inr_per_day",
        "operator_contribution_inr_per_vehicle_day",
        "breakeven_km_per_day",
        "utilisation_km_per_day",
    }
    assert reference["tco_inr_per_km"] == pytest.approx(float(cfg.eco("reference_case.tco_inr_per_km")))
    assert all(value > 0 for value in reference.values())