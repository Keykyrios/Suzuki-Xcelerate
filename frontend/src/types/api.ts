export type EvidenceLabel =
  | "SCENARIO_ASSUMPTION"
  | "SYNTHETIC"
  | "RECONSTRUCTED"
  | "MODEL_ESTIMATE"
  | "SIMULATED"
  | "OBSERVED"
  | "PUBLIC_REFERENCE"
  | "UNCALIBRATED_CONCEPT_TARGET";

export interface VehicleView {
  vehicle_id: string;
  powertrain: string;
  availability: string;
  cabin_mode: string;
  passenger_count: number;
  luggage_units: number;
  luggage_mass_kg: number;
  soc: number;
  soh: number;
  battery_temp_c: number;
  current_zone_id: string;
  state_version: number;
}

export interface ChargerView {
  charger_id: string;
  node_id: string;
  status: string;
  plug_count: number;
  plugs_in_use: number;
  rated_power_kw: number;
  site_limit_kw: number;
  tariff_inr_per_kwh: number;
  queue_length: number;
  queue_estimate_min: number;
}

export interface AssignmentView {
  vehicle_id: string;
  request_id: string;
  pickup_distance_km?: number;
  trip_distance_km?: number;
  contribution_inr?: number;
  feasible?: boolean;
}

export interface DecisionView {
  run_id: string;
  scenario_id: string;
  status: string;
  decision: {
    cabin_mode: string;
    powertrain_recommendation: string;
    route_energy_kwh: number;
    energy_kwh_per_100km: number;
    departure_soc_pct: number;
    arrival_soc_pct: number;
    reserve_soc_pct: number;
    reserve_margin_pct: number;
    expected_contribution_inr: number;
    risk_level: string;
  };
  assignments: AssignmentView[];
  reason_codes: string[];
  evidence_status: EvidenceLabel;
  stale: boolean;
}

export interface Snapshot {
  state_version: number;
  scenario_id: string;
  run_id: string | null;
  vehicles: VehicleView[];
  chargers: ChargerView[];
  requests: unknown[];
  decision: DecisionView | null;
  recommendation_stale: boolean;
  solver_status: string | null;
}
