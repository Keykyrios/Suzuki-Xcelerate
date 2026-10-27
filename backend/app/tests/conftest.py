"""Shared fixtures: configuration, physics parameter sets, route builders."""

from __future__ import annotations

from typing import Callable

import pytest

from app.config.loader import ConfigBundle, get_config
from app.schemas.charger import ChargerState
from app.schemas.common import CabinMode, ChargerStatus, PowertrainType
from app.schemas.scenario import RoadEdge
from app.schemas.vehicle import RouteResult
from app.simulation.model_config import (
    BatteryModelParams,
    HvacParams,
    NumericsParams,
    VehicleModelParams,
    battery_params,
    hvac_params,
    numerics_params,
    vehicle_model_params,
)
from app.simulation.route_energy import RouteSegment, TripInputs, build_segments, simulate_route


@pytest.fixture(scope="session")
def cfg() -> ConfigBundle:
    return get_config()


@pytest.fixture(scope="session")
def vehicle_params(cfg: ConfigBundle) -> VehicleModelParams:
    return vehicle_model_params(cfg)


@pytest.fixture(scope="session")
def hvac(cfg: ConfigBundle) -> HvacParams:
    return hvac_params(cfg)


@pytest.fixture(scope="session")
def bev_battery(cfg: ConfigBundle) -> BatteryModelParams:
    return battery_params(PowertrainType.NEXUS_E, cfg)


@pytest.fixture(scope="session")
def numerics(cfg: ConfigBundle) -> NumericsParams:
    return numerics_params(cfg)


@pytest.fixture
def edge_factory() -> Callable[..., RoadEdge]:
    def build(
        edge_id: str = "E1",
        distance_km: float = 2.0,
        speed_kmh: float = 40.0,
        grade_percent: tuple[float, ...] | list[float] = (0.0,),
        traffic_sensitivity: float = 0.4,
        from_node: str = "A",
        to_node: str = "B",
    ) -> RoadEdge:
        return RoadEdge(
            edge_id=edge_id,
            from_node=from_node,
            to_node=to_node,
            distance_km=distance_km,
            free_flow_speed_kmh=speed_kmh,
            grade_profile_percent=list(grade_percent),
            traffic_sensitivity=traffic_sensitivity,
        )

    return build


@pytest.fixture
def route_segments(edge_factory) -> Callable[..., list[RouteSegment]]:
    def build(
        distance_km: float = 3.0,
        speed_kmh: float = 45.0,
        grade_percent: tuple[float, ...] | list[float] = (0.0,),
        traffic_factor: float = 1.0,
        edge_count: int = 2,
    ) -> list[RouteSegment]:
        edges = [
            edge_factory(
                edge_id=f"E{i}",
                distance_km=distance_km,
                speed_kmh=speed_kmh,
                grade_percent=grade_percent,
                from_node=f"A{i}",
                to_node=f"A{i + 1}",
            )
            for i in range(edge_count)
        ]
        return build_segments(edges, traffic_factor)

    return build


@pytest.fixture
def trip_inputs(cfg: ConfigBundle) -> Callable[..., TripInputs]:
    def build(**overrides) -> TripInputs:
        payload = {
            "vehicle_id": "V01",
            "route_id": "route-test",
            "departure_soc": 0.8,
            "soh": 1.0,
            "ambient_temp_c": 30.0,
            "mode": CabinMode.COMMUTE,
            "occupied_zones": 2,
            "passenger_count": 3,
            "luggage_mass_kg": 20.0,
            "traffic_factor": 1.0,
            "soc_reserve_min": float(cfg.op("soc_policy.soc_reserve_min")),
            "soc_operating_min": float(cfg.op("soc_policy.soc_operating_min")),
            "config_hash": cfg.hash,
        }
        payload.update(overrides)
        return TripInputs(**payload)

    return build


@pytest.fixture
def run_route(
    vehicle_params, hvac, bev_battery, numerics, route_segments, trip_inputs, cfg
) -> Callable[..., RouteResult]:
    def run(**overrides) -> RouteResult:
        segment_overrides = {
            key: overrides.pop(key)
            for key in ("distance_km", "speed_kmh", "grade_percent", "traffic_factor", "edge_count")
            if key in overrides
        }
        segments = route_segments(**segment_overrides)
        return simulate_route(
            segments,
            vehicle_params,
            hvac,
            bev_battery,
            numerics,
            trip_inputs(**overrides),
        )

    return run


@pytest.fixture(scope="session")
def demo_scenario(cfg: ConfigBundle):
    """Deterministic demonstration world shared by scenario-level tests."""
    from app.simulation.scenario_generator import demo_spec, generate_scenario

    return generate_scenario(demo_spec(cfg), cfg)


@pytest.fixture
def charger_factory() -> Callable[..., ChargerState]:
    def build(**overrides) -> ChargerState:
        payload = {
            "charger_id": "CH01",
            "node_id": "N03",
            "status": ChargerStatus.ONLINE.value,
            "plug_count": 2,
            "plugs_in_use": 1,
            "rated_power_kw": 60.0,
            "site_limit_kw": 90.0,
            "tariff_inr_per_kwh": 12.0,
            "connectors": ["CCS2"],
            "queue_length": 0,
            "queue_estimate_min": 8.0,
            "last_updated_utc": "2026-01-01T00:00:00Z",
        }
        payload.update(overrides)
        return ChargerState.model_validate(payload)

    return build