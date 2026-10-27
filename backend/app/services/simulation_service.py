"""Simulation service: display-grade route physics and simulated actions.

Route simulations run at the configured display step (unlike the optimizer's
coarser screening step), and every vehicle-side action the demonstrator
performs is recorded as a SimulatedAction - a virtual state transition that
never touches hardware.
"""

from __future__ import annotations

from datetime import datetime, timezone

from app.config.loader import ConfigBundle, get_config
from app.schemas.common import CabinMode, CabinSource
from app.schemas.events import SimulatedAction, SimulatedActionType
from app.schemas.reason_codes import ReasonCode
from app.schemas.scenario import RouteSpec, Scenario
from app.schemas.vehicle import RouteResult, VehicleState
from app.services.state_store import StateStore, get_state_store
from app.simulation.battery import BatteryTracker
from app.simulation.city_graph import GraphModel
from app.simulation.model_config import (
    battery_params,
    hvac_params,
    numerics_params,
    vehicle_model_params,
)
from app.simulation.route_energy import TripInputs, build_segments, simulate_route


class SimulationService:
    """Runs deterministic physics and applies virtual vehicle actions."""

    def __init__(
        self,
        store: StateStore | None = None,
        cfg: ConfigBundle | None = None,
    ) -> None:
        self.store = store or get_state_store()
        self.cfg = cfg or get_config()

    def _update_vehicle(self, vehicle_id: str, updates: dict) -> VehicleState:
        """Validate the merged vehicle state before it reaches canonical state."""
        scenario = self.store.require_scenario()
        vehicle = scenario.vehicle(vehicle_id)
        if "current_zone_id" in updates:
            zones = {zone.zone_id for zone in scenario.graph.zones}
            if updates["current_zone_id"] not in zones:
                raise ValueError(f"unknown zone {updates['current_zone_id']}")
        candidate = vehicle.model_dump(mode="json")
        candidate.update(updates)
        VehicleState.model_validate(candidate)
        return self.store.update_vehicle(vehicle_id, updates)

    # -- route simulation ---------------------------------------------------
    def simulate_route(
        self,
        vehicle_id: str,
        route: RouteSpec,
        *,
        cabin_passengers: int | None = None,
        cabin_luggage_mass_kg: float | None = None,
        mode: CabinMode | None = None,
        ambient_temp_c: float | None = None,
        traffic_factor: float | None = None,
        road_grade_override_percent: float | None = None,
    ) -> RouteResult:
        """Full display-step energy/SOC simulation for one vehicle mission."""
        scenario = self.store.require_scenario()
        vehicle = scenario.vehicle(vehicle_id)
        environment = scenario.environment.model_copy(
            update={
                k: v
                for k, v in {
                    "ambient_temp_c": ambient_temp_c,
                    "traffic_factor": traffic_factor,
                }.items()
                if v is not None
            }
        )
        model = GraphModel(
            scenario.graph,
            min_speed_factor=float(self.cfg.op("traffic.min_speed_factor")),
        )
        segments = build_segments(
            [model.edges[e] for e in model.shortest_path(
                route.origin_zone, route.destination_zone, environment.traffic_factor
            ).edge_ids],
            environment.traffic_factor,
            float(self.cfg.op("traffic.min_speed_factor")),
            road_grade_override_percent,
        )
        cabin = self.store.cabin_states.get(vehicle_id)
        passengers = cabin_passengers if cabin_passengers is not None else (
            cabin.passenger_count if cabin else vehicle.passenger_count
        )
        luggage_mass = cabin_luggage_mass_kg if cabin_luggage_mass_kg is not None else (
            cabin.luggage_mass_kg if cabin else vehicle.luggage_mass_kg
        )
        cabin_mode = mode or (cabin.mode if cabin else vehicle.cabin_mode)

        result = simulate_route(
            segments,
            vehicle_model_params(self.cfg),
            hvac_params(self.cfg),
            battery_params(vehicle.powertrain, self.cfg),
            numerics_params(self.cfg),
            TripInputs(
                vehicle_id=vehicle_id,
                route_id=f"{scenario.scenario_id}-{route.origin_zone}-{route.destination_zone}",
                departure_soc=vehicle.soc,
                soh=vehicle.soh,
                ambient_temp_c=environment.ambient_temp_c,
                mode=cabin_mode,
                occupied_zones=passengers,
                passenger_count=passengers,
                luggage_mass_kg=luggage_mass,
                traffic_factor=environment.traffic_factor,
                soc_reserve_min=float(self.cfg.op("soc_policy.soc_reserve_min")),
                soc_operating_min=float(self.cfg.op("soc_policy.soc_operating_min")),
                config_hash=self.cfg.hash,
            ),
        )
        return result

    # -- virtual actions ----------------------------------------------------
    def apply_action(
        self,
        *,
        action_type: SimulatedActionType,
        target_type: str,
        target_id: str,
        requested_state: dict,
        run_id: str | None = None,
        duration_s: float = 0.0,
        originating_decision_id: str | None = None,
    ) -> SimulatedAction:
        """Apply a validated virtual action and emit its SimulatedAction record."""
        scenario = self.store.require_scenario()
        previous, resulting = self._transition(scenario, action_type, target_id, requested_state)
        action = SimulatedAction(
            action_id=f"act-{self.store.state_version + 1}",
            scenario_id=scenario.scenario_id,
            run_id=run_id or self.store.run_id,
            target_type=target_type,  # type: ignore[arg-type]
            target_id=target_id,
            action_type=action_type,
            requested_state=requested_state,
            previous_state=previous,
            resulting_state=resulting,
            simulated_at_utc=datetime.now(timezone.utc),
            duration_s=duration_s,
            reason_codes=[ReasonCode.SIMULATED_ACTION_APPLIED.value],
            originating_decision_id=originating_decision_id,
        )
        self.store.record_action(action)
        return action

    def _transition(
        self,
        scenario: Scenario,
        action_type: SimulatedActionType,
        target_id: str,
        requested: dict,
    ) -> tuple[dict, dict]:
        """Mutate canonical state for one action type; returns before/after."""
        if action_type == SimulatedActionType.CABIN_MODE_CHANGE:
            vehicle = scenario.vehicle(target_id)
            previous = {"cabin_mode": vehicle.cabin_mode.value}
            self._update_vehicle(target_id, {"cabin_mode": CabinMode(requested["mode"])})
            cabin = self.store.cabin_states.get(target_id)
            if cabin is not None:
                self.store.set_cabin_state(
                    cabin.model_copy(
                        update={
                            "mode": CabinMode(requested["mode"]),
                            "source": CabinSource.MANUAL,
                            "override_flag": True,
                        }
                    )
                )
            return previous, {"cabin_mode": CabinMode(requested["mode"]).value}

        if action_type == SimulatedActionType.SEAT_LAYOUT_CHANGE:
            vehicle = scenario.vehicle(target_id)
            previous = {"passenger_count": vehicle.passenger_count}
            count = int(requested["passenger_count"])
            self._update_vehicle(target_id, {"passenger_count": count})
            return previous, {"passenger_count": count}

        if action_type == SimulatedActionType.LUGGAGE_LAYOUT_CHANGE:
            vehicle = scenario.vehicle(target_id)
            previous = {
                "luggage_units": vehicle.luggage_units,
                "luggage_mass_kg": vehicle.luggage_mass_kg,
            }
            units = int(requested.get("luggage_units", vehicle.luggage_units))
            mass = float(requested.get("luggage_mass_kg", vehicle.luggage_mass_kg))
            self._update_vehicle(
                target_id, {"luggage_units": units, "luggage_mass_kg": mass}
            )
            return previous, {"luggage_units": units, "luggage_mass_kg": mass}

        if action_type == SimulatedActionType.VEHICLE_MOVE:
            vehicle = scenario.vehicle(target_id)
            previous = {"current_zone_id": vehicle.current_zone_id, "odometer_km": vehicle.odometer_km}
            zone_id = str(requested["zone_id"])
            distance_km = float(requested.get("distance_km", 0.0))
            updates: dict = {"current_zone_id": zone_id}
            if distance_km > 0.0:
                updates["odometer_km"] = vehicle.odometer_km + distance_km
                self._advance_battery(
                    target_id,
                    energy_kwh=float(requested.get("energy_kwh", 0.0)),
                    dt_s=float(requested.get("duration_s", 600.0)),
                )
            self._update_vehicle(target_id, updates)
            return previous, {"current_zone_id": zone_id, "odometer_km": updates.get("odometer_km", vehicle.odometer_km)}

        if action_type == SimulatedActionType.CHARGER_CONNECT:
            charger = next(c for c in scenario.chargers if c.charger_id == target_id)
            previous = {"plugs_in_use": charger.plugs_in_use, "queue_length": charger.queue_length}
            self.store.update_charger(
                target_id,
                {
                    "plugs_in_use": min(charger.plug_count, charger.plugs_in_use + 1),
                    "queue_length": max(0, charger.queue_length - 1),
                },
            )
            return previous, {"plugs_in_use": min(charger.plug_count, charger.plugs_in_use + 1)}

        if action_type == SimulatedActionType.CHARGER_DISCONNECT:
            charger = next(c for c in scenario.chargers if c.charger_id == target_id)
            previous = {"plugs_in_use": charger.plugs_in_use}
            self.store.update_charger(
                target_id, {"plugs_in_use": max(0, charger.plugs_in_use - 1)}
            )
            return previous, {"plugs_in_use": max(0, charger.plugs_in_use - 1)}

        if action_type == SimulatedActionType.HVAC_ZONE_CHANGE:
            vehicle = scenario.vehicle(target_id)
            previous = {"hvac_zones_active": vehicle.hvac_zones_active}
            zones = [str(z) for z in requested.get("zones", [])]
            self._update_vehicle(target_id, {"hvac_zones_active": zones})
            return previous, {"hvac_zones_active": zones}

        if action_type == SimulatedActionType.POWERTRAIN_SELECTION:
            vehicle = scenario.vehicle(target_id)
            previous = {"powertrain": vehicle.powertrain.value}
            # Recommendation only: the fleet's physical configuration is fixed.
            return previous, {
                "recommended": str(requested.get("recommended", vehicle.powertrain.value)),
                "applied_to_fleet": False,
            }

        if action_type == SimulatedActionType.MAINTENANCE_FLAG:
            vehicle = scenario.vehicle(target_id)
            previous = {"maintenance_flags": vehicle.maintenance_flags}
            flag = str(requested["flag"])
            flags = sorted(set(vehicle.maintenance_flags) | {flag})
            self._update_vehicle(target_id, {"maintenance_flags": flags})
            return previous, {"maintenance_flags": flags}

        raise ValueError(f"unsupported action type {action_type}")

    def _advance_battery(self, vehicle_id: str, energy_kwh: float, dt_s: float) -> None:
        tracker = self.store.battery_trackers.get(vehicle_id)
        if tracker is None or dt_s <= 0.0:
            return
        c_rate = abs(energy_kwh) / max(tracker.effective_capacity_kwh(), 1e-6) * (3600.0 / dt_s)
        tracker.advance(
            energy_delta_kwh=energy_kwh,
            dt_s=dt_s,
            temperature_c=tracker.temperature_c,
            c_rate=min(c_rate, 4.0),
        )
        self._update_vehicle(
            vehicle_id,
            {"soc": round(tracker.soc, 4), "soh": round(tracker.soh, 4)},
        )

    def battery_passport(self, vehicle_id: str) -> dict:
        """Battery passport for one vehicle from its tracker history."""
        tracker = self.store.battery_trackers.get(vehicle_id)
        if tracker is None:
            raise KeyError(f"no battery tracker for {vehicle_id}")
        replacement_cost = float(self.cfg.eco("capital.battery_replacement_inr_per_kwh"))
        return tracker.passport(replacement_cost).model_dump(mode="json")
