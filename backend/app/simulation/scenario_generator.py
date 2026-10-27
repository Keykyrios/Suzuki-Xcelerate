"""Seeded scenario generation for the starter world.

Everything a scenario contains — graph geometry, edge speeds and grades,
charger sites, fleet composition, demand records and trip requests — derives
from a seed and the configuration registry. The same seed always rebuilds the
same world, which is what makes runs replayable.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np

from app.config.loader import ConfigBundle, get_config
from app.demand.demand_model import (
    demand_context,
    generate_demand_records,
)
from app.demand.od_reconstruction import reconstruct_from_records
from app.economics.model import fare_for
from app.schemas.charger import ChargerState
from app.schemas.common import (
    CabinMode,
    ChargerStatus,
    DayType,
    EvidenceLabel,
    LuggageClass,
    PowertrainType,
    VehicleAvailability,
    utcnow,
)
from app.schemas.scenario import (
    CityGraph,
    EnvironmentControls,
    FaultInjection,
    GraphZone,
    RoadEdge,
    Scenario,
    ScenarioSpec,
    TripRequest,
)
from app.schemas.vehicle import VehicleState
from app.simulation.city_graph import GraphModel
from app.simulation.model_config import MODEL_VERSION

SCENARIO_VERSION = "1.0.0"


def build_graph(cfg: ConfigBundle | None = None) -> CityGraph:
    """Schematic city: a 4x3 zone grid plus airport connector and intercity corridor."""
    c = cfg or get_config()
    columns = int(c.demo_v("layout.zone_columns"))
    zone_count = int(c.demo_v("world.zones"))
    spacing = float(c.demo_v("layout.zone_spacing_m"))
    detour = float(c.demo_v("layout.road_detour_factor"))
    speed_base = float(c.demo_v("world.baseline_speed_m_s")) * 3.6
    speed_variation = float(c.demo_v("layout.edge_speed_variation"))
    seg_count = int(c.demo_v("layout.grade_segments_per_edge"))
    grade_min = float(c.demo_v("layout.grade_min_percent"))
    grade_max = float(c.demo_v("layout.grade_max_percent"))
    urban_sensitivity = float(c.demo_v("layout.edge_traffic_sensitivity_urban"))
    connector_sensitivity = float(c.demo_v("layout.edge_traffic_sensitivity_connector"))
    connector_speed = float(c.demo_v("layout.connector_speed_kmh"))
    airport_edge_km = float(c.demo_v("world.airport_edge_length_km"))
    corridor_km = float(c.demo_v("world.intercity_corridor_length_km"))
    toll = float(c.demo_v("layout.corridor_toll_inr"))
    prefix = str(c.demo_v("layout.zone_prefix"))

    rng = np.random.default_rng(int(c.demo_v("seeds.master_seed")))
    zones: list[GraphZone] = []
    for index in range(zone_count):
        col, row = index % columns, index // columns
        zones.append(
            GraphZone(
                zone_id=f"{prefix}{index + 1:02d}",
                name=f"{prefix}{index + 1:02d}",
                x_m=float(col * spacing),
                y_m=float(row * spacing),
                kind="urban",
            )
        )

    airport_id = str(c.demo_v("layout.airport_node_id"))
    gateway = str(c.demo_v("layout.airport_connector_zone"))
    gateway_zone = next(z for z in zones if z.zone_id == gateway)
    zones.append(
        GraphZone(
            zone_id=airport_id,
            name=airport_id,
            x_m=gateway_zone.x_m + airport_edge_km * 1000.0,
            y_m=gateway_zone.y_m,
            kind="airport",
        )
    )

    corridor_entry = str(c.demo_v("layout.corridor_entry_zone"))
    corridor_ids = [str(z) for z in c.demo_v("layout.corridor_node_ids")]
    terminus = str(c.demo_v("layout.corridor_terminus_id"))
    entry_zone = next(z for z in zones if z.zone_id == corridor_entry)
    corridor_chain = [entry_zone.zone_id, *corridor_ids, terminus]
    step_m = corridor_km * 1000.0 / (len(corridor_chain) - 1)
    for position, zone_id in enumerate(corridor_chain[1:], start=1):
        zones.append(
            GraphZone(
                zone_id=zone_id,
                name=zone_id,
                x_m=entry_zone.x_m - step_m * position,
                y_m=entry_zone.y_m,
                kind="corridor",
                is_intercity_terminal=zone_id == terminus,
            )
        )

    def grades() -> list[float]:
        return [float(rng.uniform(grade_min, grade_max)) for _ in range(seg_count)]

    edges: list[RoadEdge] = []

    def add_edge(a: str, b: str, distance_km: float, speed_kmh: float,
                 sensitivity: float, edge_toll: float = 0.0) -> None:
        edges.append(
            RoadEdge(
                edge_id=f"E-{a}-{b}",
                from_node=a,
                to_node=b,
                distance_km=round(distance_km, 3),
                free_flow_speed_kmh=round(speed_kmh, 2),
                grade_profile_percent=[round(g, 2) for g in grades()],
                traffic_sensitivity=sensitivity,
                toll_inr=edge_toll,
                bidirectional=True,
            )
        )

    for index, zone in enumerate(zones[:zone_count]):
        col, row = index % columns, index // columns
        if col + 1 < columns:
            neighbour = zones[index + 1]
            speed = speed_base * float(1.0 + rng.uniform(-speed_variation, speed_variation))
            add_edge(zone.zone_id, neighbour.zone_id, spacing * detour / 1000.0, speed, urban_sensitivity)
        if index + columns < zone_count:
            neighbour = zones[index + columns]
            speed = speed_base * float(1.0 + rng.uniform(-speed_variation, speed_variation))
            add_edge(zone.zone_id, neighbour.zone_id, spacing * detour / 1000.0, speed, urban_sensitivity)

    add_edge(gateway, airport_id, airport_edge_km, connector_speed, connector_sensitivity)
    for a, b in zip(corridor_chain, corridor_chain[1:]):
        # One toll plaza on the mid-corridor segment, charged per traversal.
        segment_toll = toll if (a in corridor_ids and b in corridor_ids) else 0.0
        add_edge(a, b, step_m / 1000.0, connector_speed, connector_sensitivity, segment_toll)

    return CityGraph(graph_id="nexus7-starter-graph", version=SCENARIO_VERSION, zones=zones, edges=edges)


def build_chargers(
    graph: CityGraph, seed: int, cfg: ConfigBundle | None = None
) -> list[ChargerState]:
    """Charging sites placed on the zones configured for them."""
    c = cfg or get_config()
    rng = np.random.default_rng(seed)
    zone_ids = {z.zone_id for z in graph.zones}
    session_min = float(c.op("service.charger_plug_service_time_s")) / 60.0
    max_queue = int(c.demo_v("charger_sites.initial_queue_max"))
    shoulder = float(c.eco("electricity.shoulder_inr_per_kwh"))

    sites = []
    for parameter in c.demo.all_parameters():
        if not parameter.path.startswith("charger_sites."):
            continue
        if "." in parameter.path[len("charger_sites."):]:
            continue
        extras = parameter.__pydantic_extra__ or {}
        if "node_zone" not in extras:
            continue  # scalar settings such as initial_queue_max
        node_zone = str(parameter.node_zone)
        if node_zone not in zone_ids:
            raise ValueError(f"{parameter.path}: node zone {node_zone} is not in the graph")
        queue = int(rng.integers(0, max_queue + 1))
        sites.append(
            ChargerState(
                charger_id=str(parameter.value),
                node_id=node_zone,
                status=ChargerStatus.ONLINE,
                plug_count=int(parameter.plug_count),
                plugs_in_use=int(rng.integers(0, min(int(parameter.plug_count), queue) + 1)),
                rated_power_kw=float(parameter.rated_power_kw),
                site_limit_kw=float(parameter.site_limit_kw),
                tariff_inr_per_kwh=shoulder,
                connectors=[str(x) for x in parameter.connectors],
                queue_length=queue,
                queue_estimate_min=round(session_min, 1),
                last_updated_utc=utcnow(),
            )
        )
    sites.sort(key=lambda ch: ch.charger_id)
    return sites


def build_fleet(
    graph: CityGraph, seed: int, cfg: ConfigBundle | None = None
) -> list[VehicleState]:
    """Ten-vehicle fleet with configured composition, SOH band and start states."""
    c = cfg or get_config()
    rng = np.random.default_rng(seed + 1)
    count = int(c.demo_v("world.vehicles"))
    e_count = int(c.demo_v("fleet.nexus_e_count"))
    x_count = int(c.demo_v("fleet.nexus_x_count"))
    if e_count + x_count != count:
        raise ValueError(
            f"fleet composition {e_count}+{x_count} does not match world.vehicles={count}"
        )
    id_prefix = str(c.demo_v("fleet.id_prefix"))
    soh_lo, soh_hi = [float(v) for v in c.demo_v("uncertainty_families.fleet_soh_range")]
    soc_lo = float(c.demo_v("fleet.initial_soc_min"))
    soc_hi = float(c.demo_v("fleet.initial_soc_max"))
    soc_lo = max(soc_lo, float(c.op("soc_policy.soc_operating_min")))
    soc_hi = min(soc_hi, float(c.op("soc_policy.soc_operating_max")))
    odometer = float(c.demo_v("fleet.odometer_start_km"))
    charge_limit = float(c.demo_v("fleet.charge_limit_soc"))
    temp_c = float(c.demo_v("fleet.battery_temp_initial_c"))
    urban = [z for z in graph.zones if z.kind == "urban"]
    focus_id = str(c.demo_v("demo_script.focus_vehicle_id"))
    focus_zone = str(c.demo_v("layout.focus_initial_zone"))
    demo_mode = CabinMode(str(c.demo_v("demo_script.initial_cabin_mode")))
    demo_soc = float(c.demo_v("demo_script.initial_soc"))

    powertrains = [PowertrainType.NEXUS_E] * e_count + [PowertrainType.NEXUS_X] * x_count
    rng.shuffle(powertrains)

    vehicles: list[VehicleState] = []
    for index in range(count):
        vehicle_id = f"{id_prefix}{index + 1:02d}"
        is_focus = vehicle_id == focus_id
        zone = (
            focus_zone
            if is_focus
            else str(urban[int(rng.integers(0, len(urban)))].zone_id)
        )
        mode = demo_mode if is_focus else CabinMode(list(CabinMode)[int(rng.integers(0, 4))])
        vehicles.append(
            VehicleState(
                vehicle_id=vehicle_id,
                powertrain=powertrains[index],
                availability=VehicleAvailability.AVAILABLE,
                cabin_mode=mode,
                passenger_count=0,
                luggage_units=0,
                luggage_mass_kg=0.0,
                soc=demo_soc if is_focus else float(rng.uniform(soc_lo, soc_hi)),
                soh=float(rng.uniform(soh_lo, soh_hi)),
                battery_temp_c=temp_c,
                current_zone_id=zone,
                current_edge_id=None,
                next_available_time_s=0.0,
                odometer_km=odometer,
                charge_limit_soc=charge_limit,
                hvac_zones_active=[],
                maintenance_flags=[],
                model_version=MODEL_VERSION,
                state_timestamp_utc=utcnow(),
                state_version=0,
            )
        )
    return vehicles


def request_pool(
    graph: CityGraph, seed: int, count: int, cfg: ConfigBundle | None = None
) -> list[TripRequest]:
    """Trip requests drawn from the seeded demand history of the starter world."""
    c = cfg or get_config()
    rng = np.random.default_rng(seed + 2)
    records = generate_demand_records(graph.zones, int(c.demo_v("world.demand_records")), seed, c)
    model = GraphModel(graph)

    weights = np.array([r.request_count + 1.0 for r in records], dtype=float)
    weights /= weights.sum()
    span_s = float(c.demo_v("world_defaults.request_time_span_s"))
    window_s = float(c.op("service.pickup_window_s"))
    pax_min = int(c.demo_v("request_defaults.passengers_min"))
    pax_max = int(c.demo_v("request_defaults.passengers_max"))
    luggage_p = float(c.demo_v("request_defaults.luggage_probability"))
    luggage_max = int(c.demo_v("request_defaults.luggage_units_max"))
    large_share = float(c.demo_v("request_defaults.large_luggage_share"))
    small_mass = float(c.vehicle.parameter("luggage_classes.small").default_mass_kg)
    large_mass = float(c.vehicle.parameter("luggage_classes.large").default_mass_kg)

    requests: list[TripRequest] = []
    attempts = 0
    while len(requests) < count and attempts < count * 20:
        attempts += 1
        record = records[int(rng.choice(len(records), p=weights))]
        if record.origin_zone_id == record.destination_zone_id:
            continue
        path = model.shortest_path(
            record.origin_zone_id, record.destination_zone_id, traffic_factor=1.0
        )
        service = record.service_class
        mode_record = c.vehicle.parameter(f"cabin_modes.{service.value.lower()}")
        mode_limit = int(mode_record.max_passengers)
        luggage_limit = int(mode_record.max_luggage_units)
        allowed_classes = set(mode_record.luggage_class_allowed)
        passengers = int(rng.integers(pax_min, min(pax_max, mode_limit) + 1))
        luggage_units = 0
        large_units = 0
        if rng.random() < luggage_p:
            luggage_units = int(
                rng.integers(1, min(luggage_max, luggage_limit) + 1)
            )
            if "large" in allowed_classes:
                large_units = int(round(luggage_units * large_share))
        luggage_mass = large_units * large_mass + (luggage_units - large_units) * small_mass
        classes: dict[LuggageClass, int] = {}
        if large_units:
            classes[LuggageClass.LARGE] = large_units
        if luggage_units - large_units:
            classes[LuggageClass.SMALL] = luggage_units - large_units
        fare = fare_for(service, passengers, path.distance_km, c)
        request_time = float(rng.uniform(0.0, span_s))
        requests.append(
            TripRequest(
                request_id=f"TR-{len(requests) + 1:03d}",
                pickup_zone_id=record.origin_zone_id,
                dropoff_zone_id=record.destination_zone_id,
                request_time_s=round(request_time, 1),
                pickup_window_start_s=round(request_time, 1),
                pickup_window_end_s=round(request_time + window_s, 1),
                passenger_count=passengers,
                luggage_units=luggage_units,
                luggage_classes=classes,
                luggage_mass_kg=round(luggage_mass, 1),
                service_class=service,
                expected_fare_inr=fare.total_inr,
                cancellation_penalty_inr=float(c.opt("dispatch_weights.late_pickup_penalty_inr")),
            )
        )
    return requests


def named_fault(name: str, target_id: str | None = None, cfg: ConfigBundle | None = None) -> FaultInjection:
    """Look up a fault in the configured injection catalog."""
    c = cfg or get_config()
    path = f"fault_injections.{name}"
    parameter = c.demo.parameter(path)
    return FaultInjection(
        fault_id=f"F-{name}",
        name=name,
        target=str(parameter.target),
        parameter=str(parameter.parameter),
        value=parameter.value,
        target_id=target_id,
    )


def faults_from_spec(spec: ScenarioSpec, cfg: ConfigBundle | None = None) -> list[FaultInjection]:
    """Parse the scenario fault payload: either a named fault or key/value pairs."""
    c = cfg or get_config()
    if not spec.fault_injection:
        return []
    payload = dict(spec.fault_injection)
    if "name" in payload:
        return [named_fault(str(payload["name"]), payload.get("target_id"), c)]

    faults: list[FaultInjection] = []
    if "charger_id" in payload and "status" in payload:
        fault = named_fault("charger_offline", str(payload["charger_id"]), c)
        faults.append(fault.model_copy(update={"value": payload["status"]}))
    if "ambient_temp_c" in payload:
        fault = named_fault("heat_wave", spec.vehicle_id, c)
        faults.append(fault.model_copy(update={"value": float(payload["ambient_temp_c"])}))
    if "traffic_factor" in payload:
        fault = named_fault("traffic_spike", None, c)
        faults.append(fault.model_copy(update={"value": float(payload["traffic_factor"])}))
    if "demand_multiplier" in payload:
        fault = named_fault("demand_surge", None, c)
        faults.append(fault.model_copy(update={"value": float(payload["demand_multiplier"])}))
    if "soh" in payload and "vehicle_id" in payload:
        fault = named_fault("low_soh_vehicle", str(payload["vehicle_id"]), c)
        faults.append(fault.model_copy(update={"value": float(payload["soh"])}))
    if not faults:
        raise ValueError(
            "fault_injection payload does not match a supported fault: "
            f"{sorted(payload)}"
        )
    return faults


def apply_faults(scenario: Scenario, faults: list[FaultInjection]) -> Scenario:
    """Apply fault injections to a scenario by their configured target/parameter."""
    if not faults:
        return scenario
    chargers = [ch.model_dump() for ch in scenario.chargers]
    vehicles = [v.model_dump() for v in scenario.vehicles]
    environment = scenario.environment.model_copy()
    demand_context_obj = scenario.demand_context

    for fault in faults:
        if fault.target == "charger":
            if fault.target_id is None:
                raise ValueError(f"{fault.name} requires a target charger id")
            matched = False
            for index, charger in enumerate(chargers):
                if charger["charger_id"] == fault.target_id:
                    chargers[index] = {**charger, "status": fault.value}
                    matched = True
            if not matched:
                raise ValueError(f"{fault.name}: unknown charger {fault.target_id}")
        elif fault.target == "scenario":
            if fault.parameter == "ambient_temp_c":
                environment = environment.model_copy(
                    update={"ambient_temp_c": float(fault.value)}
                )
            elif fault.parameter == "traffic_factor":
                environment = environment.model_copy(
                    update={"traffic_factor": float(fault.value)}
                )
            elif fault.parameter == "demand_multiplier":
                environment = environment.model_copy(
                    update={"demand_multiplier": float(fault.value)}
                )
                demand_context_obj = demand_context_obj.model_copy(
                    update={"demand_multiplier": float(fault.value)}
                )
            else:
                raise ValueError(f"{fault.name}: unsupported scenario parameter {fault.parameter}")
        elif fault.target == "vehicle":
            if fault.parameter != "soh":
                raise ValueError(f"{fault.name}: unsupported vehicle parameter {fault.parameter}")
            target = fault.target_id
            if target is None:
                target = min(vehicles, key=lambda v: v["soh"])["vehicle_id"]
            matched = False
            for index, vehicle in enumerate(vehicles):
                if vehicle["vehicle_id"] == target:
                    vehicles[index] = {**vehicle, "soh": float(fault.value)}
                    matched = True
            if not matched:
                raise ValueError(f"{fault.name}: unknown vehicle {target}")
        else:
            raise ValueError(f"{fault.name}: unknown fault target {fault.target}")

    return scenario.model_copy(
        update={
            "chargers": [ChargerState.model_validate(ch) for ch in chargers],
            "vehicles": [VehicleState.model_validate(v) for v in vehicles],
            "environment": environment,
            "demand_context": demand_context_obj,
            "faults": faults,
        }
    )


def _od_flows_from_records(graph: CityGraph, seed: int, cfg: ConfigBundle) -> list:
    records = generate_demand_records(graph.zones, int(cfg.demo_v("world.demand_records")), seed, cfg)
    zone_ids = [z.zone_id for z in graph.zones]
    index = {zid: i for i, zid in enumerate(zone_ids)}
    origins = np.zeros(len(zone_ids))
    dests = np.zeros(len(zone_ids))
    for record in records:
        origins[index[record.origin_zone_id]] += record.request_count
        dests[index[record.destination_zone_id]] += record.request_count
    if origins.sum() <= 0:
        return []
    # Marginals must balance for a coherent flow matrix.
    total_o, total_d = origins.sum(), dests.sum()
    dests *= total_o / max(total_d, 1e-9)
    result = reconstruct_from_records(graph.zones, origins, dests, cfg)
    return result.to_flows()


def generate_scenario(
    spec: ScenarioSpec,
    cfg: ConfigBundle | None = None,
    created_utc: datetime | None = None,
) -> Scenario:
    """Build the canonical, serializable, replayable world for a scenario request."""
    c = cfg or get_config()
    graph = build_graph(c)
    seed = spec.seed
    chargers = build_chargers(graph, seed, c)
    vehicles = build_fleet(graph, seed, c)

    if spec.vehicle_id is not None and not any(v.vehicle_id == spec.vehicle_id for v in vehicles):
        raise ValueError(f"unknown vehicle {spec.vehicle_id}")

    if spec.cabin is not None and spec.vehicle_id is not None:
        target = next(v for v in vehicles if v.vehicle_id == spec.vehicle_id)
        index = vehicles.index(target)
        vehicles[index] = target.model_copy(
            update={
                "cabin_mode": spec.cabin.mode,
                "passenger_count": spec.cabin.passenger_count,
                "luggage_units": spec.cabin.luggage_count,
                "luggage_mass_kg": spec.cabin.luggage_mass_kg,
            }
        )

    environment = EnvironmentControls(
        ambient_temp_c=float(c.demo_v("world_defaults.ambient_temp_c")),
        traffic_factor=float(c.demo_v("world_defaults.traffic_factor")),
        demand_multiplier=1.0,
    )
    if spec.environment is not None:
        environment = spec.environment

    if spec.requests_override is not None:
        requests = list(spec.requests_override)
    else:
        requests = request_pool(
            graph, seed, int(c.demo_v("world_defaults.active_requests")), c
        )

    if spec.route is not None:
        known = {z.zone_id for z in graph.zones}
        for zone_id in (spec.route.origin_zone, spec.route.destination_zone):
            if zone_id not in known:
                raise ValueError(f"route references unknown zone {zone_id}")

    faults = faults_from_spec(spec, c)
    start_hour = int(c.demo_v("world_defaults.horizon_start_hour"))
    context = demand_context(
        start_hour,
        DayType.WEEKDAY,
        environment.demand_multiplier,
        c,
    )

    scenario = Scenario(
        scenario_id=spec.scenario_id,
        seed=seed,
        graph=graph,
        vehicles=vehicles,
        chargers=chargers,
        requests=requests,
        environment=environment,
        demand_context=context,
        od_flows=_od_flows_from_records(graph, seed, c),
        faults=[],
        tariffs_inr_per_kwh={
            "peak": float(c.eco("electricity.peak_inr_per_kwh")),
            "shoulder": float(c.eco("electricity.shoulder_inr_per_kwh")),
            "offpeak": float(c.eco("electricity.offpeak_inr_per_kwh")),
            "dc_fast_surcharge": float(c.eco("electricity.dc_fast_surcharge_inr_per_kwh")),
        },
        created_utc=created_utc or utcnow(),
        scenario_version=SCENARIO_VERSION,
        provenance=EvidenceLabel.SYNTHETIC,
    )
    return apply_faults(scenario, faults)


def demo_spec(cfg: ConfigBundle | None = None) -> ScenarioSpec:
    """The deterministic resettable demonstration scenario."""
    c = cfg or get_config()
    focus = str(c.demo_v("demo_script.focus_vehicle_id"))
    return ScenarioSpec(
        scenario_id=str(c.demo_v("demo_script.scenario_id")),
        seed=int(c.demo_v("seeds.master_seed")),
        vehicle_id=focus,
        cabin=None,
        environment=None,
        route=None,
        fault_injection=None,
    )


def sample_uncertainty_families(seed: int, cfg: ConfigBundle | None = None) -> dict:
    """Draw one realization of the five configured uncertainty families."""
    c = cfg or get_config()
    rng = np.random.default_rng(seed)
    demand_sigma = float(c.demo_v("uncertainty_families.demand_volume_sigma"))
    travel_sigma = float(c.demo_v("uncertainty_families.travel_time_sigma"))
    outage_p = float(c.demo_v("uncertainty_families.charger_outage_probability"))
    queue_sigma = float(c.demo_v("uncertainty_families.charger_queue_sigma"))
    temp_lo, temp_hi = [float(v) for v in c.demo_v("uncertainty_families.ambient_temp_range_c")]
    soh_lo, soh_hi = [float(v) for v in c.demo_v("uncertainty_families.fleet_soh_range")]
    return {
        "demand_multiplier": float(np.clip(1.0 + rng.normal(0.0, demand_sigma), 0.2, 4.0)),
        "traffic_factor": float(np.clip(1.0 + rng.normal(0.0, travel_sigma), 0.8, 2.5)),
        "charger_outages": rng.random(int(c.demo_v("world.chargers"))) < outage_p,
        "queue_delta": rng.normal(0.0, queue_sigma),
        "ambient_temp_c": float(rng.uniform(temp_lo, temp_hi)),
        "vehicle_soh": [float(rng.uniform(soh_lo, soh_hi)) for _ in range(int(c.demo_v("world.vehicles")))],
    }


def generate_scenario_batch(
    base_spec: ScenarioSpec,
    seeds: list[int],
    cfg: ConfigBundle | None = None,
) -> list[Scenario]:
    """Batch of scenarios across the uncertainty families, one per seed."""
    c = cfg or get_config()
    scenarios: list[Scenario] = []
    for seed in seeds:
        spec = base_spec.model_copy(
            update={"seed": seed, "scenario_id": f"{base_spec.scenario_id}_s{seed}"}
        )
        scenario = generate_scenario(spec, c)
        drawn = sample_uncertainty_families(seed, c)
        chargers = [ch.model_dump() for ch in scenario.chargers]
        for index, offline in enumerate(drawn["charger_outages"]):
            if offline:
                chargers[index] = {**chargers[index], "status": ChargerStatus.OFFLINE.value}
            chargers[index] = {
                **chargers[index],
                "queue_length": max(
                    0, int(round(chargers[index]["queue_length"] + drawn["queue_delta"]))
                ),
            }
        vehicles = [v.model_dump() for v in scenario.vehicles]
        for index, soh in enumerate(drawn["vehicle_soh"]):
            vehicles[index] = {**vehicles[index], "soh": round(min(soh, vehicles[index]["soh"]), 4)}
        environment = scenario.environment.model_copy(
            update={
                "ambient_temp_c": round(drawn["ambient_temp_c"], 1),
                "traffic_factor": round(drawn["traffic_factor"], 2),
                "demand_multiplier": round(drawn["demand_multiplier"], 2),
            }
        )
        scenarios.append(
            scenario.model_copy(
                update={
                    "chargers": [ChargerState.model_validate(ch) for ch in chargers],
                    "vehicles": [VehicleState.model_validate(v) for v in vehicles],
                    "environment": environment,
                    "demand_context": scenario.demand_context.model_copy(
                        update={"demand_multiplier": environment.demand_multiplier}
                    ),
                }
            )
        )
    return scenarios


def demand_records_for(scenario: Scenario, cfg: ConfigBundle | None = None):
    """Regenerate the scenario's demand history deterministically from its seed."""
    c = cfg or get_config()
    return generate_demand_records(
        scenario.graph.zones, int(c.demo_v("world.demand_records")), scenario.seed, c
    )


def fallback_service_class(zones: list[GraphZone], origin: str) -> CabinMode:
    """Service class implied by the zone kind of a request's origin."""
    kind = next((z.kind for z in zones if z.zone_id == origin), "urban")
    if kind == "airport":
        return CabinMode.AIRPORT
    if kind == "corridor":
        return CabinMode.INTERCITY_PREMIUM
    return CabinMode.COMMUTE


__all__ = [
    "SCENARIO_VERSION",
    "apply_faults",
    "build_chargers",
    "build_fleet",
    "build_graph",
    "demo_spec",
    "demand_records_for",
    "faults_from_spec",
    "generate_scenario",
    "generate_scenario_batch",
    "named_fault",
    "request_pool",
    "sample_uncertainty_families",
]
