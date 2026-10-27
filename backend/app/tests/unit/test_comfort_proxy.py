"""Comfort proxy: bounded display aid that improves with HVAC effort."""

from __future__ import annotations

from app.schemas.common import CabinMode
from app.simulation.thermal import comfort_proxy


def test_proxy_stays_inside_its_declared_display_band(hvac):
    extreme = comfort_proxy(hvac, CabinMode.POOL, 8, 55.0, hvac_kw=0.0)
    freezing = comfort_proxy(hvac, CabinMode.POOL, 0, -20.0, hvac_kw=hvac.base_load_kw * 4)
    assert -1.0 <= extreme <= 1.0
    assert -1.0 <= freezing <= 1.0


def test_more_hvac_improves_hot_ambient_comfort(hvac):
    uncooled = comfort_proxy(hvac, CabinMode.AIRPORT, 4, 42.0, hvac_kw=0.0)
    cooled = comfort_proxy(hvac, CabinMode.AIRPORT, 4, 42.0, hvac_kw=hvac.base_load_kw * 3)
    assert cooled < uncooled


def test_occupancy_beyond_pairs_penalises_comfort(hvac):
    light = comfort_proxy(hvac, CabinMode.COMMUTE, 2, 30.0, hvac_kw=hvac.base_load_kw)
    crowded = comfort_proxy(hvac, CabinMode.COMMUTE, 6, 30.0, hvac_kw=hvac.base_load_kw)
    assert crowded > light
