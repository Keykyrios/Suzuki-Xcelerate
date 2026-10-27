import { useCallback, useEffect, useState } from "react";
import { ValueChip } from "../components/ValueChip";
import { CameraPanel } from "../components/CameraPanel";
import { useLiveSnapshot } from "../state/live";
import type { EvidenceLabel } from "../types/api";

interface CabinStateView {
  mode: string;
  passenger_count: number;
  luggage_units: number;
  confidence: number;
  source: string;
  override_flag: boolean;
  proposed_mode: string | null;
  mode_proposal_reason: string | null;
}

interface VisionStatus {
  camera_enabled: boolean;
  camera_source: string;
  detector_available: boolean;
  detector_error: string | null;
  model_version: string | null;
  confidence_threshold: number;
  cabin_states: Record<string, CabinStateView>;
}

interface EconomicsReport {
  currency: string;
  cost_structure: Record<string, unknown>;
  tariff: Record<string, unknown>;
  reference_case: Record<string, unknown>;
  tco: Record<string, unknown>;
  fleet_day: Record<string, unknown>;
}

const FAULTS = ["heat_wave", "traffic_spike", "demand_surge", "low_soh_vehicle"] as const;

async function post(path: string, body?: unknown): Promise<Response> {
  return fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body ?? {}),
  });
}

function fmt(value: number | null | undefined, digits = 1): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return value.toFixed(digits);
}

export function App() {
  const { snapshot, connection, refresh } = useLiveSnapshot();
  const [economics, setEconomics] = useState<EconomicsReport | null>(null);
  const [vision, setVision] = useState<VisionStatus | null>(null);
  const [visionVehicle, setVisionVehicle] = useState("NX-01");
  const [busy, setBusy] = useState(false);
  const [lastError, setLastError] = useState<string | null>(null);

  const loadEconomics = useCallback(() => {
    void fetch("/api/economics")
      .then((r) => r.json())
      .then((data: EconomicsReport) => setEconomics(data))
      .catch(() => setEconomics(null));
  }, []);

  const loadVision = useCallback(() => {
    void fetch("/api/vision/state")
      .then((r) => r.json())
      .then((data: VisionStatus) => setVision(data))
      .catch(() => setVision(null));
  }, []);

  useEffect(loadEconomics, [loadEconomics]);
  useEffect(loadVision, [loadVision]);

  const act = useCallback(
    async (fn: () => Promise<Response>) => {
      setBusy(true);
      setLastError(null);
      try {
        const response = await fn();
        if (!response.ok) {
          const detail = await response.text();
          throw new Error(`${response.status}: ${detail.slice(0, 160)}`);
        }
        refresh();
        loadEconomics();
        loadVision();
      } catch (error) {
        setLastError(error instanceof Error ? error.message : String(error));
      } finally {
        setBusy(false);
      }
    },
    [refresh, loadEconomics, loadVision],
  );

  const decision = snapshot?.decision ?? null;
  const headline = decision?.decision ?? null;

  return (
    <div className="app">
      <header className="topbar">
        <h1>NEXUS-7 MOBILITY DEMONSTRATOR</h1>
        <span className={`badge ${connection === "live" ? "live" : "offline"}`}>
          {connection === "live" ? "LIVE STATE STREAM" : connection.toUpperCase()}
        </span>
        <span className="badge">scenario {snapshot?.scenario_id ?? "—"}</span>
        <span className="badge">run {snapshot?.run_id ?? "none"}</span>
        <span className="badge">state v{snapshot?.state_version ?? 0}</span>
        {snapshot?.recommendation_stale && (
          <span className="badge warn">RECOMMENDATION STALE — RE-RUN OPTIMIZATION</span>
        )}
      </header>

      {lastError && <p className="badge offline">action failed: {lastError}</p>}

      <div className="grid">
        <section className="panel" data-testid="controls">
          <h2>Scenario controls</h2>
          <div className="controls">
            <button disabled={busy} onClick={() => act(() => post("/api/demo/reset"))}>
              Reset demo
            </button>
            <button disabled={busy} onClick={() => act(() => post("/api/optimize", {}))}>
              Optimize fleet
            </button>
          </div>
          <h2>Fault injections</h2>
          <div className="controls">
            {FAULTS.map((fault) => (
              <button
                key={fault}
                disabled={busy}
                onClick={() =>
                  act(() =>
                    post("/api/scenarios", {
                      scenario_id: `${snapshot?.scenario_id ?? "demo"}-${fault}`,
                      seed: snapshot ? 42 : 0,
                      fault_injection: { name: fault },
                    }),
                  )
                }
              >
                {fault.replace("_", " ")}
              </button>
            ))}
            <button
              className="danger"
              disabled={busy || !snapshot?.chargers.length}
              onClick={() => {
                const charger = snapshot?.chargers[0];
                if (!charger) return;
                return act(() =>
                  post(`/api/chargers/${charger.charger_id}/status`, {
                    status: charger.status === "OFFLINE" ? "ONLINE" : "OFFLINE",
                  }),
                );
              }}
            >
              Toggle first charger outage
            </button>
          </div>
        </section>

        <section className="panel" data-testid="vehicles">
          <h2>Fleet state</h2>
          {(snapshot?.vehicles ?? []).map((vehicle) => (
            <div className="row" key={vehicle.vehicle_id}>
              <span>
                {vehicle.vehicle_id} · {vehicle.cabin_mode} · {vehicle.powertrain}
              </span>
              <span className="muted">
                SOC {(vehicle.soc * 100).toFixed(0)}% · SOH {(vehicle.soh * 100).toFixed(0)}% ·{" "}
                {vehicle.passenger_count} pax
              </span>
              <button
                disabled={busy}
                onClick={() =>
                  act(() =>
                    post(`/api/vehicles/${vehicle.vehicle_id}/state`, {
                      passenger_count: Math.min(8, vehicle.passenger_count + 1),
                    }),
                  )
                }
              >
                +pax
              </button>
              <button
                disabled={busy || vehicle.passenger_count === 0}
                onClick={() =>
                  act(() =>
                    post(`/api/vehicles/${vehicle.vehicle_id}/state`, {
                      passenger_count: Math.max(0, vehicle.passenger_count - 1),
                    }),
                  )
                }
              >
                −pax
              </button>
            </div>
          ))}
        </section>

        <section className="panel" data-testid="chargers">
          <h2>Chargers</h2>
          {(snapshot?.chargers ?? []).map((charger) => (
            <div className="row" key={charger.charger_id}>
              <span>{charger.charger_id}</span>
              <span className="muted">
                {charger.rated_power_kw} kW · site {charger.site_limit_kw} kW · queue{" "}
                {charger.queue_length} · {charger.tariff_inr_per_kwh} INR/kWh
              </span>
              <span className={`status-pill ${charger.status.toLowerCase()}`}>
                {charger.status}
              </span>
            </div>
          ))}
        </section>

        <section className="panel" data-testid="vision">
          <h2>Cabin sensing (on-device vision)</h2>
          {!vision && <p className="muted">Vision status unavailable.</p>}
          {vision && (
            <>
              <div className="row">
                <span>
                  camera {vision.camera_enabled ? "ON" : "OFF"} · source {vision.camera_source} ·{" "}
                  {vision.model_version ?? "detector unavailable"}
                </span>
                <span className="muted">confidence threshold {vision.confidence_threshold}</span>
              </div>
              {vision.detector_error && (
                <p className="badge offline">{vision.detector_error}</p>
              )}
              <div className="row">
                <label htmlFor="vision-vehicle">target vehicle</label>
                <select
                  id="vision-vehicle"
                  value={visionVehicle}
                  onChange={(event) => setVisionVehicle(event.target.value)}
                >
                  {(snapshot?.vehicles ?? []).map((vehicle) => (
                    <option key={vehicle.vehicle_id} value={vehicle.vehicle_id}>
                      {vehicle.vehicle_id}
                    </option>
                  ))}
                </select>
              </div>
              {vision.camera_enabled && (
                <CameraPanel vehicleId={visionVehicle} onResult={loadVision} />
              )}
              {(snapshot?.vehicles ?? []).slice(0, 4).map((vehicle) => {
                const cabin = vision.cabin_states?.[vehicle.vehicle_id];
                return (
                  <div className="row" key={`vision-${vehicle.vehicle_id}`}>
                    <span>
                      {vehicle.vehicle_id} · {cabin ? cabin.mode : "no reading"} ·{" "}
                      {cabin ? `${cabin.passenger_count} pax / ${cabin.luggage_units} bags` : "—"}
                      {cabin && (
                        <span className="muted">
                          {" "}
                          · {cabin.source.toLowerCase()} · conf {(cabin.confidence * 100).toFixed(0)}%
                          {cabin.override_flag ? " · override" : ""}
                        </span>
                      )}
                    </span>
                    <button
                      disabled={busy || !vision.camera_enabled}
                      onClick={() => act(() => post(`/api/vision/capture/${vehicle.vehicle_id}`))}
                    >
                      Capture
                    </button>
                    <button
                      disabled={busy}
                      onClick={() =>
                        act(() =>
                          post(`/api/vision/override/${vehicle.vehicle_id}`, {
                            mode: "AIRPORT",
                            passenger_count: 4,
                            small_luggage: 1,
                            large_luggage: 3,
                          }),
                        )
                      }
                    >
                      Manual 4 pax / airport
                    </button>
                  </div>
                );
              })}
              <div className="controls">
                <button
                  disabled={busy}
                  onClick={() =>
                    act(() => post("/api/vision/camera", { enabled: !vision.camera_enabled }))
                  }
                >
                  {vision.camera_enabled ? "Turn camera off" : "Turn camera on"}
                </button>
              </div>
            </>
          )}
        </section>

        <section className="panel" data-testid="decision">
          <h2>Decision</h2>
          {!headline && <p className="muted">Run an optimization to produce a decision.</p>}
          {headline && (
            <>
              <div className="row">
                <span>
                  {decision?.status} · {assignmentsLabel(decision?.assignments.length ?? 0)} served
                </span>
                <span className="muted">evidence: {decision?.evidence_status}</span>
              </div>
              <div className="chips">
                <ValueChip
                  label="Route energy"
                  unit="kWh"
                  evidence={(decision?.evidence_status ?? "MODEL_ESTIMATE") as EvidenceLabel}
                >
                  {fmt(headline.route_energy_kwh)}
                </ValueChip>
                <ValueChip label="Energy intensity" unit="kWh/100km" evidence="MODEL_ESTIMATE">
                  {fmt(headline.energy_kwh_per_100km)}
                </ValueChip>
                <ValueChip label="Departure SOC" unit="%" evidence="MODEL_ESTIMATE">
                  {fmt(headline.departure_soc_pct)}
                </ValueChip>
                <ValueChip label="Arrival SOC" unit="%" evidence="MODEL_ESTIMATE">
                  {fmt(headline.arrival_soc_pct)}
                </ValueChip>
                <ValueChip label="Reserve margin" unit="%" evidence="SCENARIO_ASSUMPTION">
                  {fmt(headline.reserve_margin_pct)}
                </ValueChip>
                <ValueChip label="Expected contribution" unit="INR" evidence="SCENARIO_ASSUMPTION">
                  {fmt(headline.expected_contribution_inr)}
                </ValueChip>
              </div>
              <ul className="reasons">
                {(decision?.reason_codes ?? []).slice(0, 5).map((code, index) => (
                  <li key={`${code}-${index}`}>{code.replaceAll("_", " ")}</li>
                ))}
              </ul>
            </>
          )}
        </section>

        <section className="panel" data-testid="economics">
          <h2>Economics</h2>
          {!economics && <p className="muted">Economics report unavailable.</p>}
          {economics && (
            <div className="row">
              <span>currency {economics.currency}</span>
              <span className="muted">
                tariffs and cost structure loaded from configuration ·{" "}
                {Object.keys(economics.fleet_day ?? {}).length} fleet-day fields
              </span>
            </div>
          )}
        </section>
      </div>
    </div>
  );
}

function assignmentsLabel(count: number): string {
  return `${count}`;
}
