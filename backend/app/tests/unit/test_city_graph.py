"""City graph: route validity, traffic scaling, missing edges and unknown zones."""

from __future__ import annotations

import pytest

from app.schemas.reason_codes import ReasonCode
from app.schemas.scenario import CityGraph, GraphZone, RoadEdge
from app.simulation.city_graph import GraphModel, RoutePlanningError, effective_speed_kmh


def test_demo_graph_meets_the_required_shape(demo_scenario, cfg):
    graph = demo_scenario.graph
    assert len(graph.zones) >= 12
    assert len(demo_scenario.chargers) >= 3
    assert any(z.kind == "airport" for z in graph.zones)
    assert any(z.is_intercity_terminal for z in graph.zones)
    fleet_size = int(cfg.demo_v("fleet.nexus_e_count")) + int(
        cfg.demo_v("fleet.nexus_x_count")
    )
    assert len(demo_scenario.vehicles) == fleet_size
    assert fleet_size == 10


def test_every_edge_references_existing_nodes(demo_scenario):
    zone_ids = {z.zone_id for z in demo_scenario.graph.zones}
    for edge in demo_scenario.graph.edges:
        assert edge.from_node in zone_ids
        assert edge.to_node in zone_ids
        assert edge.distance_km > 0.0
        assert edge.free_flow_speed_kmh > 0.0
        assert edge.traffic_sensitivity >= 0.0
        assert edge.grade_profile_percent


def test_graph_constructs_and_validates(demo_scenario):
    model = GraphModel(demo_scenario.graph)
    assert len(model.zones) == len(demo_scenario.graph.zones)
    assert len(model.edges) == len(demo_scenario.graph.edges)


def test_paths_only_use_existing_edges(demo_scenario):
    model = GraphModel(demo_scenario.graph)
    zones = list(model.zones)
    for origin in zones[:4]:
        for destination in zones[-4:]:
            if origin == destination:
                continue
            path = model.shortest_path(origin, destination, 1.0)
            assert path.edge_ids
            assert all(edge_id in model.edges for edge_id in path.edge_ids)
            assert path.distance_km > 0.0
            assert path.travel_time_min > 0.0


def test_path_ends_at_the_requested_destination(demo_scenario):
    model = GraphModel(demo_scenario.graph)
    zones = list(model.zones)
    origin, destination = zones[0], zones[-1]
    path = model.shortest_path(origin, destination, 1.0)
    first_edge = model.edges[path.edge_ids[0]]
    last_edge = model.edges[path.edge_ids[-1]]
    assert first_edge.from_node == origin
    assert last_edge.to_node == destination


def test_unknown_zone_is_rejected_with_a_reason_code(demo_scenario):
    model = GraphModel(demo_scenario.graph)
    with pytest.raises(RoutePlanningError) as excinfo:
        model.shortest_path("NOPE", list(model.zones)[0], 1.0)
    assert excinfo.value.reason_code == ReasonCode.ROUTE_ZONE_UNKNOWN


def test_identical_origin_and_destination_is_rejected(demo_scenario):
    model = GraphModel(demo_scenario.graph)
    zone = list(model.zones)[0]
    with pytest.raises(RoutePlanningError):
        model.shortest_path(zone, zone, 1.0)


def test_disconnected_graph_is_rejected(edge_factory):
    zones = [
        GraphZone(zone_id="A", name="A", x_m=0.0, y_m=0.0),
        GraphZone(zone_id="B", name="B", x_m=1000.0, y_m=0.0),
        GraphZone(zone_id="ISLAND", name="Island", x_m=5000.0, y_m=5000.0),
    ]
    edges = [
        edge_factory(edge_id="E1", from_node="A", to_node="B", distance_km=1.0, speed_kmh=40.0),
    ]
    with pytest.raises(RoutePlanningError) as excinfo:
        GraphModel(CityGraph(graph_id="g", version="1", zones=zones, edges=edges))
    assert excinfo.value.reason_code == ReasonCode.ROUTE_EDGE_MISSING


def test_edge_referencing_a_missing_node_is_rejected(edge_factory):
    zones = [GraphZone(zone_id="A", name="A", x_m=0.0, y_m=0.0)]
    edges = [edge_factory(edge_id="E1", from_node="A", to_node="GHOST")]
    with pytest.raises(RoutePlanningError) as excinfo:
        GraphModel(CityGraph(graph_id="g", version="1", zones=zones, edges=edges))
    assert excinfo.value.reason_code == ReasonCode.ROUTE_EDGE_MISSING


def test_to_path_rejects_edge_not_in_the_graph(demo_scenario, edge_factory, graph=None):
    model = GraphModel(demo_scenario.graph)
    stranger = edge_factory(edge_id="STRANGER", from_node="A", to_node="B", distance_km=1.0)
    with pytest.raises(RoutePlanningError) as excinfo:
        model.to_path("A", "B", [stranger], 1.0)
    assert excinfo.value.reason_code == ReasonCode.ROUTE_EDGE_MISSING


def test_empty_route_is_rejected_before_any_physics_runs():
    from app.schemas.reason_codes import ReasonCode as Codes
    from app.simulation.route_energy import PhysicsInputError, build_segments

    with pytest.raises(PhysicsInputError) as excinfo:
        build_segments([], 1.0)
    assert excinfo.value.reason_code == Codes.ROUTE_EDGE_MISSING


def test_traffic_slows_the_adjusted_speed(edge_factory):
    edge = edge_factory(speed_kmh=50.0, traffic_sensitivity=1.0)
    assert effective_speed_kmh(edge, 1.0) == pytest.approx(50.0)
    slow = effective_speed_kmh(edge, 1.5)
    assert slow < 50.0
    assert slow == pytest.approx(50.0 / (1.0 + 1.0 * 0.5))


def test_traffic_speed_is_floored_by_the_configured_minimum(edge_factory, cfg):
    edge = edge_factory(speed_kmh=60.0, traffic_sensitivity=1.5)
    floor = float(cfg.op("traffic.min_speed_factor"))
    fast = effective_speed_kmh(edge, 0.8, floor)
    assert fast >= 60.0 * floor - 1e-9
    assert effective_speed_kmh(edge, 3.0, 0.0) <= effective_speed_kmh(edge, 3.0, floor) + 1e-9


def test_path_duration_matches_the_sum_of_edge_durations(demo_scenario):
    model = GraphModel(demo_scenario.graph)
    zones = list(model.zones)
    path = model.shortest_path(zones[1], zones[-2], 1.2)
    total = sum(model.edge_time_min(model.edges[e], 1.2) for e in path.edge_ids)
    assert path.travel_time_min == pytest.approx(total, rel=1e-9)


def test_mean_grade_is_distance_weighted(demo_scenario):
    model = GraphModel(demo_scenario.graph)
    zones = list(model.zones)
    path = model.shortest_path(zones[0], zones[-1], 1.0)
    edges = [model.edges[e] for e in path.edge_ids]
    weight = sum(e.distance_km for e in edges)
    expected = sum(e.mean_grade_percent * e.distance_km for e in edges) / weight
    assert path.mean_grade_percent == pytest.approx(expected, rel=1e-9)


def test_road_edge_rejects_grade_outside_the_supported_band(edge_factory):
    with pytest.raises(Exception):
        edge_factory(grade_percent=(15.0,))
    with pytest.raises(Exception):
        edge_factory(distance_km=0.0)


def test_bidirectional_edges_link_both_directions(edge_factory):
    zones = [
        GraphZone(zone_id="A", name="A", x_m=0.0, y_m=0.0),
        GraphZone(zone_id="B", name="B", x_m=1000.0, y_m=0.0),
    ]
    edges = [edge_factory(edge_id="E1", from_node="A", to_node="B", distance_km=1.0)]
    model = GraphModel(CityGraph(graph_id="g", version="1", zones=zones, edges=edges))
    assert model.shortest_path("B", "A", 1.0).edge_ids == ("E1",)


def test_edge_schema_requires_unique_id_and_positive_fields(edge_factory):
    with pytest.raises(Exception):
        RoadEdge(
            edge_id="E1",
            from_node="A",
            to_node="B",
            distance_km=-1.0,
            free_flow_speed_kmh=30.0,
            grade_profile_percent=[0.0],
            traffic_sensitivity=0.5,
        )