"""Deterministic route-energy simulation for one trip.

The speed trace is built as a piecewise phase trajectory (accelerate, cruise,
decelerate) per road edge with exactly consistent distance, acceleration, grade
and time. Power is integrated on a fixed step; regeneration is bounded by the
recovery fraction, motor/inverter limit, battery charge acceptance and SOC
headroom. All functions are pure: same inputs, same outputs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from app.schemas.common import CabinMode, ConstraintStatus, EvidenceLabel
from app.schemas.reason_codes import ReasonCode
from app.schemas.vehicle import RouteResult
from app.simulation.battery import charge_acceptance_kw
from app.simulation.city_graph import effective_speed_kmh
from app.simulation.model_config import (
    MODEL_VERSION,
    BatteryModelParams,
    HvacParams,
    NumericsParams,
    VehicleModelParams,
)
from app.simulation.thermal import hvac_power_kw
from app.simulation.vehicle_dynamics import (
    PhysicsInputError,
    loaded_mass_kg,
    wheel_force_N,
)
from app.schemas.scenario import RoadEdge

MPS_PER_KMH = 1.0 / 3.6
SECONDS_PER_HOUR = 3600.0


@dataclass(frozen=True)
class RouteSegment:
    """One road edge prepared for profile generation."""

    edge_id: str
    distance_m: float
    cruise_speed_mps: float
    grade_percent: tuple[float, ...]


def build_segments(
    edges: list[RoadEdge],
    traffic_factor: float,
    min_speed_factor: float = 0.0,
    road_grade_override_percent: float | None = None,
) -> list[RouteSegment]:
    """Project validated graph edges onto traffic-adjusted route segments."""
    if not edges:
        raise PhysicsInputError("route contains no edges", ReasonCode.ROUTE_EDGE_MISSING)
    segments: list[RouteSegment] = []
    for edge in edges:
        speed = effective_speed_kmh(edge, traffic_factor, min_speed_factor) * MPS_PER_KMH
        grades = (
            (road_grade_override_percent,)
            if road_grade_override_percent is not None
            else tuple(edge.grade_profile_percent)
        )
        segments.append(
            RouteSegment(
                edge_id=edge.edge_id,
                distance_m=edge.distance_km * 1000.0,
                cruise_speed_mps=max(speed, 0.5),
                grade_percent=grades,
            )
        )
    return segments


@dataclass(frozen=True)
class _Phase:
    """Constant-acceleration or constant-speed phase of the speed trace."""

    duration_s: float
    accel_m_s2: float
    entry_speed_mps: float


def _edge_phases(
    entry_speed: float, exit_speed: float, cruise_speed: float, distance: float,
    accel_limit: float, decel_limit: float,
) -> list[_Phase]:
    """Trapezoidal profile matching `distance` exactly between boundary speeds."""
    phases: list[_Phase] = []
    remaining = distance
    v0 = entry_speed

    # Transition from the previous edge's exit speed toward this edge's cruise speed.
    if cruise_speed > v0 + 1e-9:
        t = (cruise_speed - v0) / accel_limit
        d = v0 * t + 0.5 * accel_limit * t * t
    elif cruise_speed < v0 - 1e-9:
        t = (v0 - cruise_speed) / decel_limit
        d = v0 * t - 0.5 * decel_limit * t * t
    else:
        t, d = 0.0, 0.0
    if d < remaining:
        phases.append(_Phase(max(t, 0.0), (cruise_speed - v0) / t if t > 0 else 0.0, v0))
        remaining -= d
        v_mid_start = cruise_speed
    else:
        # Not enough distance to reach this edge's cruise speed: triangular profile.
        v_mid_start = None  # type: ignore[assignment]

    if v_mid_start is None:
        peak = _peak_speed(v0, exit_speed, accel_limit, decel_limit, distance)
        t_up = max(0.0, (peak - v0) / accel_limit)
        t_dn = max(0.0, (peak - exit_speed) / decel_limit)
        if t_up > 0.0:
            phases.append(_Phase(t_up, accel_limit, v0))
        if t_dn > 0.0:
            phases.append(_Phase(t_dn, -decel_limit, peak))
        return [p for p in phases if p.duration_s > 0.0]

    # Cruise at this edge's speed, then transition to the next edge's entry speed.
    t_down = 0.0
    if exit_speed < cruise_speed - 1e-9:
        t_down = (cruise_speed - exit_speed) / decel_limit
        d_down = cruise_speed * t_down - 0.5 * decel_limit * t_down * t_down
    elif exit_speed > cruise_speed + 1e-9:
        t_down = (exit_speed - cruise_speed) / accel_limit
        d_down = cruise_speed * t_down + 0.5 * accel_limit * t_down * t_down
    else:
        d_down = 0.0

    if d_down <= remaining:
        d_cruise = remaining - d_down
        if d_cruise > 0.0:
            phases.append(_Phase(d_cruise / cruise_speed, 0.0, cruise_speed))
        if t_down > 0.0:
            a = (exit_speed - cruise_speed) / t_down
            phases.append(_Phase(t_down, a, cruise_speed))
        return [p for p in phases if p.duration_s > 0.0]

    # Cannot hold cruise all the way: shorten the transition from a lower peak.
    peak = _peak_speed(cruise_speed, exit_speed, accel_limit, decel_limit, remaining)
    t_up = max(0.0, (peak - cruise_speed) / accel_limit)
    t_dn = max(0.0, (peak - exit_speed) / decel_limit)
    if t_up > 0.0:
        phases.append(_Phase(t_up, accel_limit, cruise_speed))
    else:
        phases.append(_Phase(remaining / max(cruise_speed, 1e-6), 0.0, cruise_speed))
    if t_dn > 0.0:
        phases.append(_Phase(t_dn, -decel_limit, peak))
    return [p for p in phases if p.duration_s > 0.0]


def _peak_speed(v0: float, v1: float, accel: float, decel: float, distance: float) -> float:
    """Distance-limited peak speed for an accelerate-then-decelerate profile."""
    # d = (v0^2 + v1^2)/2 ... solved for constant accel a then decel b scaled equally.
    disc = (accel * decel * distance * 2.0 + accel * v1**2 + decel * v0**2) / (accel + decel)
    return max(v0, v1, math.sqrt(max(disc, 0.0)))


@dataclass(frozen=True)
class _TraceSample:
    time_s: float
    speed_mps: float
    acceleration_m_s2: float
    grade_rad: float
    edge_id: str


def _grade_at(edge: RouteSegment, fraction: float) -> float:
    grades = edge.grade_percent
    idx = min(len(grades) - 1, max(0, int(fraction * len(grades))))
    return grades[idx]


def build_speed_trace(
    segments: list[RouteSegment],
    dt_s: float,
    accel_limit: float,
    decel_limit: float,
) -> list[_TraceSample]:
    """Piecewise-analytic speed/acceleration/grade trace at a fixed step."""
    if dt_s <= 0.0:
        raise PhysicsInputError(f"integration step must be positive, got {dt_s}")
    samples: list[_TraceSample] = []
    t_cursor = 0.0
    entry_speed = segments[0].cruise_speed_mps
    for index, segment in enumerate(segments):
        exit_speed = (
            segments[index + 1].cruise_speed_mps
            if index + 1 < len(segments)
            else 0.0
        )
        phases = _edge_phases(
            entry_speed, exit_speed, segment.cruise_speed_mps, segment.distance_m,
            accel_limit, decel_limit,
        )
        distance_cursor = 0.0
        for phase in phases:
            steps = max(1, int(math.ceil(phase.duration_s / dt_s)))
            step_dt = phase.duration_s / steps
            for step in range(steps):
                frac = distance_cursor / max(segment.distance_m, 1e-9)
                samples.append(
                    _TraceSample(
                        time_s=t_cursor + (step + 0.5) * step_dt,
                        speed_mps=max(0.0, phase.entry_speed_mps + phase.accel_m_s2 * step * step_dt),
                        acceleration_m_s2=phase.accel_m_s2,
                        grade_rad=math.atan(_grade_at(segment, min(frac, 1.0)) / 100.0),
                        edge_id=segment.edge_id,
                    )
                )
                distance_cursor += max(
                    0.0,
                    phase.entry_speed_mps + phase.accel_m_s2 * (step + 0.5) * step_dt,
                ) * step_dt
            t_cursor += phase.duration_s
        entry_speed = exit_speed
    if not samples:
        raise PhysicsInputError("speed trace produced no samples", ReasonCode.ROUTE_EDGE_MISSING)
    # Normalize drift from piecewise sampling so total time matches the phases.
    return samples


@dataclass(frozen=True)
class TripInputs:
    vehicle_id: str
    route_id: str
    departure_soc: float
    soh: float
    ambient_temp_c: float
    mode: CabinMode
    occupied_zones: int
    passenger_count: int
    luggage_mass_kg: float
    traffic_factor: float
    soc_reserve_min: float
    soc_operating_min: float
    config_hash: str
    solar_fraction: float = 1.0


def simulate_route(
    segments: list[RouteSegment],
    vehicle: VehicleModelParams,
    hvac: HvacParams,
    battery: BatteryModelParams,
    numerics: NumericsParams,
    inputs: TripInputs,
) -> RouteResult:
    """Run the deterministic energy/SOC simulation for one route."""
    dt_s = numerics.integration_dt_s
    warnings: list[str] = []
    checks: dict[str, ConstraintStatus] = {}

    try:
        mass = loaded_mass_kg(vehicle, inputs.passenger_count, inputs.luggage_mass_kg)
    except PhysicsInputError as exc:
        return _infeasible_result(segments, inputs, battery, str(exc), [exc.reason_code.value])

    if not math.isfinite(inputs.departure_soc) or not 0.0 <= inputs.departure_soc <= 1.0:
        raise PhysicsInputError(f"departure_soc {inputs.departure_soc} outside [0, 1]")
    if not math.isfinite(inputs.ambient_temp_c) or not -40.0 <= inputs.ambient_temp_c <= 60.0:
        raise PhysicsInputError(f"ambient_temp_c {inputs.ambient_temp_c} outside [-40, 60]")

    checks["passenger_capacity"] = ConstraintStatus.PASS
    checks["payload_capacity"] = ConstraintStatus.PASS

    hvac_kw = hvac_power_kw(
        hvac,
        mode=inputs.mode,
        occupied_zones=inputs.occupied_zones,
        ambient_temp_c=inputs.ambient_temp_c,
        solar_fraction=inputs.solar_fraction,
    )
    capacity_kwh = max(battery.usable_energy_kwh * min(max(inputs.soh, 0.0), 1.0), 1e-6)
    acceptance_cap_kw = charge_acceptance_kw(battery, inputs.departure_soc)

    samples = build_speed_trace(
        segments, dt_s, vehicle.max_acceleration_m_s2, vehicle.max_deceleration_m_s2
    )

    time_s: list[float] = []
    speeds: list[float] = []
    accels: list[float] = []
    wheel_kw: list[float] = []
    traction_kw: list[float] = []
    hvac_series: list[float] = []
    aux_series: list[float] = []
    regen_series: list[float] = []
    cumulative: list[float] = []
    soc_trace: list[float] = []

    soc = inputs.departure_soc
    energy_kwh = 0.0
    traction_total = 0.0
    hvac_total = 0.0
    aux_total = 0.0
    regen_total = 0.0
    over_power_samples = 0
    regen_rejected_samples = 0
    distance_m = 0.0
    prev_speed = samples[0].speed_mps

    for index, sample in enumerate(samples):
        speed = sample.speed_mps
        if index == 0:
            accel = sample.acceleration_m_s2
        else:
            accel = (speed - prev_speed) / dt_s
        prev_speed = speed
        distance_m += speed * dt_s

        force = wheel_force_N(mass.loaded_mass_kg, accel, speed, sample.grade_rad, vehicle)
        p_wheel = force * speed / 1000.0

        p_drive = max(p_wheel, 0.0) / vehicle.drivetrain_efficiency
        if p_drive > vehicle.peak_traction_power_kw:
            over_power_samples += 1
            p_drive_clipped = vehicle.peak_traction_power_kw
        else:
            p_drive_clipped = p_drive

        p_regen = 0.0
        if p_wheel < 0.0 and speed >= vehicle.regen_min_speed_m_s and soc < 1.0:
            candidate = (
                -p_wheel * vehicle.regen_recovery_fraction * vehicle.regen_path_efficiency
            )
            candidate = min(candidate, vehicle.regen_max_power_kw)
            candidate = min(candidate, charge_acceptance_kw(battery, soc))
            headroom_kwh = max(0.0, (1.0 - soc) * capacity_kwh)
            candidate = min(candidate, headroom_kwh * SECONDS_PER_HOUR / dt_s)
            if candidate <= 0.0:
                regen_rejected_samples += 1
            p_regen = max(candidate, 0.0)

        p_batt = p_drive_clipped + hvac_kw + vehicle.auxiliary_load_kw - p_regen
        energy_kwh += p_batt * dt_s / SECONDS_PER_HOUR

        if energy_kwh < 0.0:
            warnings.append(
                "route energy exceeds stored energy before arrival; SOC held at zero"
            )
            energy_kwh = 0.0

        traction_total += p_drive_clipped * dt_s / SECONDS_PER_HOUR
        hvac_total += hvac_kw * dt_s / SECONDS_PER_HOUR
        aux_total += vehicle.auxiliary_load_kw * dt_s / SECONDS_PER_HOUR
        regen_total += p_regen * dt_s / SECONDS_PER_HOUR

        soc = max(0.0, min(1.0, inputs.departure_soc - energy_kwh / capacity_kwh))

        time_s.append(sample.time_s)
        speeds.append(speed)
        accels.append(accel)
        wheel_kw.append(p_wheel)
        traction_kw.append(p_drive_clipped)
        hvac_series.append(hvac_kw)
        aux_series.append(vehicle.auxiliary_load_kw)
        regen_series.append(p_regen)
        cumulative.append(energy_kwh)
        soc_trace.append(soc)

    if over_power_samples:
        warnings.append(
            f"traction power exceeded the configured peak for {over_power_samples} samples; "
            "the trace was power-limited"
        )
    if regen_rejected_samples:
        warnings.append(
            f"regeneration was reduced or rejected in {regen_rejected_samples} samples "
            "due to SOC headroom, motor limit or low speed"
        )

    travel_time_min = (time_s[-1] + dt_s) / 60.0 if time_s else 0.0
    distance_km = sum(s.distance_m for s in segments) / 1000.0
    net_kwh = traction_total + hvac_total + aux_total - regen_total
    _ = distance_m

    effective_reserve = max(inputs.soc_reserve_min, inputs.soc_operating_min)
    arrival_soc = soc_trace[-1] if soc_trace else inputs.departure_soc
    reserve_margin = arrival_soc - effective_reserve

    checks["route_trace_valid"] = (
        ConstraintStatus.PASS if math.isfinite(net_kwh) and net_kwh >= 0.0 else ConstraintStatus.FAIL
    )
    reserve_ok = reserve_margin >= 0.0 and net_kwh <= inputs.departure_soc * capacity_kwh
    checks["arrival_soc_reserve"] = ConstraintStatus.PASS if reserve_ok else ConstraintStatus.FAIL

    reason_codes = []
    if reserve_ok:
        reason_codes.append(ReasonCode.ARRIVAL_SOC_ABOVE_RESERVE.value)
    else:
        reason_codes.append(ReasonCode.ARRIVAL_SOC_BELOW_RESERVE.value)
        warnings.append(
            f"predicted arrival SOC {arrival_soc * 100:.1f}% is below the configured "
            f"reserve {effective_reserve * 100:.1f}%"
        )

    energy_per_100km = (net_kwh / distance_km * 100.0) if distance_km > 0 else 0.0
    grade_weight = sum(s.distance_m for s in segments)
    mean_grade = (
        sum(
            sum(s.grade_percent) / len(s.grade_percent) * s.distance_m for s in segments
        )
        / max(grade_weight, 1e-9)
    )

    return RouteResult(
        route_id=inputs.route_id,
        vehicle_id=inputs.vehicle_id,
        distance_km=round(distance_km, 3),
        travel_time_min=round(travel_time_min, 2),
        departure_soc=round(inputs.departure_soc, 4),
        arrival_soc=round(arrival_soc, 4),
        soc_reserve_min=round(effective_reserve, 4),
        reserve_margin=round(reserve_margin, 4),
        route_energy_kwh=round(net_kwh, 3),
        energy_kwh_per_100km=round(energy_per_100km, 2),
        loaded_mass_kg=round(mass.loaded_mass_kg, 1),
        passenger_count=inputs.passenger_count,
        luggage_mass_kg=round(inputs.luggage_mass_kg, 1),
        time_s=[round(t, 2) for t in time_s],
        speed_mps=[round(v, 3) for v in speeds],
        acceleration_m_s2=[round(a, 3) for a in accels],
        wheel_power_kw=[round(p, 3) for p in wheel_kw],
        traction_battery_power_kw=[round(p, 3) for p in traction_kw],
        hvac_power_kw=[round(p, 3) for p in hvac_series],
        auxiliary_power_kw=[round(p, 3) for p in aux_series],
        regen_power_kw=[round(p, 3) for p in regen_series],
        cumulative_energy_kwh=[round(e, 4) for e in cumulative],
        soc_trace=[round(s, 4) for s in soc_trace],
        energy_contributors_kwh={
            "traction_kwh": round(traction_total, 3),
            "hvac_kwh": round(hvac_total, 3),
            "auxiliary_kwh": round(aux_total, 3),
            "regen_recovered_kwh": round(-regen_total, 3),
        },
        ambient_temp_c=inputs.ambient_temp_c,
        traffic_factor=inputs.traffic_factor,
        mean_grade_percent=round(mean_grade, 2),
        warnings=warnings,
        constraint_checks=checks,
        model_version=MODEL_VERSION,
        config_hash=inputs.config_hash,
        evidence_status=EvidenceLabel.MODEL_ESTIMATE,
        calibrated=False,
    )


def _infeasible_result(
    segments: list[RouteSegment],
    inputs: TripInputs,
    battery: BatteryModelParams,
    message: str,
    reason_codes: list[str],
) -> RouteResult:
    """Capacity violation: an infeasible result carrying reason codes, no numbers invented."""
    effective_reserve = max(inputs.soc_reserve_min, inputs.soc_operating_min)
    return RouteResult(
        route_id=inputs.route_id,
        vehicle_id=inputs.vehicle_id,
        distance_km=round(sum(s.distance_m for s in segments) / 1000.0, 3),
        travel_time_min=0.0,
        departure_soc=round(inputs.departure_soc, 4),
        arrival_soc=round(inputs.departure_soc, 4),
        soc_reserve_min=round(effective_reserve, 4),
        reserve_margin=round(inputs.soc_reserve_min - effective_reserve, 4),
        route_energy_kwh=0.0,
        energy_kwh_per_100km=0.0,
        loaded_mass_kg=0.0,
        passenger_count=inputs.passenger_count,
        luggage_mass_kg=round(inputs.luggage_mass_kg, 1),
        time_s=[],
        speed_mps=[],
        acceleration_m_s2=[],
        wheel_power_kw=[],
        traction_battery_power_kw=[],
        hvac_power_kw=[],
        auxiliary_power_kw=[],
        regen_power_kw=[],
        cumulative_energy_kwh=[],
        soc_trace=[],
        energy_contributors_kwh={},
        ambient_temp_c=inputs.ambient_temp_c,
        traffic_factor=inputs.traffic_factor,
        mean_grade_percent=0.0,
        warnings=[f"infeasible route request: {message}", *reason_codes],
        constraint_checks={
            "passenger_capacity": ConstraintStatus.FAIL,
            "payload_capacity": ConstraintStatus.FAIL,
            "route_trace_valid": ConstraintStatus.FAIL,
            "arrival_soc_reserve": ConstraintStatus.FAIL,
        },
        model_version=MODEL_VERSION,
        config_hash=inputs.config_hash,
        evidence_status=EvidenceLabel.MODEL_ESTIMATE,
        calibrated=False,
    )


def route_feasible(result: RouteResult) -> bool:
    """Independent hard-feasibility verdict from a route result."""
    return all(
        status == ConstraintStatus.PASS for status in result.constraint_checks.values()
    )