"""Scenario lifecycle: create, load, reset, fault, and persist scenarios."""

from __future__ import annotations

from app.config.loader import ConfigBundle, get_config
from app.data_access.repository import Repository, demand_record_rows
from app.schemas.common import ChargerStatus
from app.schemas.scenario import Scenario, ScenarioSpec
from app.schemas.vehicle import VehicleState
from app.services.state_store import StateStore, get_state_store
from app.simulation.scenario_generator import (
    apply_faults,
    demand_records_for,
    demo_spec,
    faults_from_spec,
    generate_scenario,
)


class ScenarioService:
    """Owns scenario creation and the store's scenario lifecycle."""

    def __init__(
        self,
        store: StateStore | None = None,
        cfg: ConfigBundle | None = None,
        repository: Repository | None = None,
    ) -> None:
        self.store = store or get_state_store()
        self.cfg = cfg or get_config()
        self.repository = repository or Repository(self.cfg.root)

    def create(self, spec: ScenarioSpec, run_id: str | None = None) -> Scenario:
        """Validate, build, persist, and activate a scenario."""
        scenario = generate_scenario(spec, self.cfg)
        self._persist(scenario)
        self.store.load_scenario(scenario, run_id=run_id)
        return scenario

    def reset_demo(self, run_id: str | None = None) -> Scenario:
        """Restore the deterministic demonstration scenario."""
        return self.create(demo_spec(self.cfg), run_id=run_id)

    def set_charger_status(
        self, charger_id: str, status: ChargerStatus, queue_length: int | None = None
    ) -> Scenario:
        """Change a simulated charger's availability; invalidates decisions."""
        updates: dict[str, object] = {"status": status}
        if queue_length is not None:
            scenario = self.store.require_scenario()
            charger = next(c for c in scenario.chargers if c.charger_id == charger_id)
            updates["queue_length"] = max(0, queue_length)
            updates["plugs_in_use"] = min(charger.plug_count, max(0, queue_length))
        self.store.update_charger(charger_id, updates)
        self.store.set_stale(True, reason=f"charger {charger_id} changed to {status.value}")
        return self.store.require_scenario()

    def apply_fault(self, fault_name: str, target_id: str | None = None) -> Scenario:
        """Apply a named fault from the configured injection catalog."""
        scenario = self.store.require_scenario()
        fault = faults_from_spec(
            ScenarioSpec(
                scenario_id=scenario.scenario_id,
                seed=scenario.seed,
                fault_injection={"name": fault_name, "target_id": target_id},
            ),
            self.cfg,
        )
        updated = apply_faults(scenario, fault)
        self.store.load_scenario(updated, run_id=self.store.run_id)
        return updated

    def update_vehicle_state(self, vehicle_id: str, updates: dict[str, object]) -> VehicleState:
        """Permitted simulated vehicle state updates through one gate."""
        permitted = {
            "availability",
            "cabin_mode",
            "passenger_count",
            "luggage_units",
            "luggage_mass_kg",
            "soc",
            "soh",
            "battery_temp_c",
            "current_zone_id",
            "current_edge_id",
            "next_available_time_s",
            "charge_limit_soc",
            "hvac_zones_active",
            "maintenance_flags",
            "odometer_km",
        }
        rejected = sorted(set(updates) - permitted)
        if rejected:
            raise ValueError(f"updates not permitted for vehicle state: {rejected}")
        vehicle = self.store.update_vehicle(vehicle_id, updates)
        self.store.set_stale(True, reason="vehicle state changed")
        return vehicle

    def _persist(self, scenario: Scenario) -> None:
        self.repository.write_json_document(
            f"scenario_{scenario.scenario_id}", scenario.model_dump(mode="json")
        )
        records = demand_records_for(scenario, self.cfg)
        self.repository.write_records(
            "demand_records", demand_record_rows(records), synthetic=True
        )
