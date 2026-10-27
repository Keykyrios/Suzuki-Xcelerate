"""HVAC load: occupancy, ambient temperature, mode factors, comfort proxy."""

from __future__ import annotations

import pytest

from app.schemas.common import CabinMode
from app.simulation.thermal import comfort_proxy, hvac_power_kw, updated_battery_temp_c


def test_more_occupied_zones_draw_more_power(hvac):
    powers = [hvac_power_kw(hvac, CabinMode.COMMUTE, z, 22.0, solar_fraction=0.0) for z in range(0, 5)]
    assert powers == sorted(powers)
    mode_step = hvac.per_occupied_zone_kw * hvac.mode_factors[CabinMode.COMMUTE.value.lower()]
    # Steps above the configured floor follow the mode factor exactly; the
    # first step may be lifted by the floor clamp.
    for earlier, later in zip(powers[1:], powers[2:]):
        assert later - earlier == pytest.approx(mode_step, rel=1e-9)


def test_zone_increment_matches_configuration(hvac):
    one = hvac_power_kw(hvac, CabinMode.COMMUTE, 1, hvac.ambient_reference_c, solar_fraction=0.0)
    three = hvac_power_kw(hvac, CabinMode.COMMUTE, 3, hvac.ambient_reference_c, solar_fraction=0.0)
    mode_step = 2 * hvac.per_occupied_zone_kw * hvac.mode_factors[CabinMode.COMMUTE.value.lower()]
    assert three - one == pytest.approx(mode_step, rel=1e-9)


def test_hotter_ambient_increases_load(hvac):
    mild = hvac_power_kw(hvac, CabinMode.COMMUTE, 2, 25.0)
    hot = hvac_power_kw(hvac, CabinMode.COMMUTE, 2, 45.0)
    assert hot >= mild


def test_power_is_clamped_into_the_configured_band(hvac):
    hottest = hvac_power_kw(hvac, CabinMode.POOL, 8, 55.0, solar_fraction=1.0)
    assert hottest <= hvac.base_load_kw * hvac.ambient_max_factor + 1e-9
    coldest = hvac_power_kw(hvac, CabinMode.POOL, 0, -10.0, solar_fraction=0.0)
    assert coldest >= hvac.base_load_kw - 1e-9


def test_mode_factor_scales_the_load(hvac):
    airport = hvac_power_kw(hvac, CabinMode.AIRPORT, 3, hvac.ambient_reference_c)
    commute = hvac_power_kw(hvac, CabinMode.COMMUTE, 3, hvac.ambient_reference_c)
    factor = hvac.mode_factors[CabinMode.AIRPORT.value.lower()] / hvac.mode_factors[
        CabinMode.COMMUTE.value.lower()
    ]
    assert airport / commute == pytest.approx(factor, rel=0.1)


def test_hvac_energy_in_a_route_grows_with_ambient_temperature(run_route):
    mild = run_route(ambient_temp_c=25.0, distance_km=4.0)
    hot = run_route(ambient_temp_c=45.0, distance_km=4.0)
    assert (
        hot.energy_contributors_kwh["hvac_kwh"]
        >= mild.energy_contributors_kwh["hvac_kwh"]
    )
    assert hot.route_energy_kwh >= mild.route_energy_kwh
    assert hot.arrival_soc <= mild.arrival_soc + 1e-9


def test_more_open_zones_increase_route_hvac_energy(run_route):
    few = run_route(occupied_zones=1, ambient_temp_c=40.0, distance_km=4.0)
    many = run_route(occupied_zones=4, ambient_temp_c=40.0, distance_km=4.0)
    assert (
        many.energy_contributors_kwh["hvac_kwh"]
        >= few.energy_contributors_kwh["hvac_kwh"]
    )


def test_comfort_proxy_stays_inside_its_display_range(hvac):
    for ambient in (-5.0, 10.0, 25.0, 40.0, 55.0):
        for zones in (0, 2, 6):
            value = comfort_proxy(hvac, CabinMode.AIRPORT, zones, ambient, hvac.base_load_kw)
            assert -1.0 <= value <= 1.0


def test_comfort_proxy_improves_with_more_conditioning_when_hot(hvac):
    ambient = hvac.comfort_temp_reference_c + 7.0
    warm = comfort_proxy(hvac, CabinMode.COMMUTE, 2, ambient, hvac.base_load_kw)
    cooled = comfort_proxy(hvac, CabinMode.COMMUTE, 2, ambient, hvac.base_load_kw * 4.0)
    assert -1.0 <= cooled < warm <= 1.0


def test_battery_temperature_relaxes_toward_ambient(bev_battery):
    start = 60.0
    cooled = updated_battery_temp_c(bev_battery, start, 25.0, internal_loss_kw=0.0, dt_s=600.0)
    assert 25.0 < cooled < start
    settle = start
    for _ in range(500):
        settle = updated_battery_temp_c(bev_battery, settle, 25.0, 0.0, 600.0)
    assert settle == pytest.approx(25.0, abs=0.05)


def test_internal_losses_raise_battery_temperature(bev_battery):
    passive = updated_battery_temp_c(bev_battery, 30.0, 30.0, 0.0, 300.0)
    loaded = updated_battery_temp_c(bev_battery, 30.0, 30.0, 20.0, 300.0)
    assert loaded > passive