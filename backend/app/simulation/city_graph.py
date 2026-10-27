"""City graph model, validation, and deterministic pathfinding."""

from __future__ import annotations

import heapq
from dataclasses import dataclass

from app.schemas.reason_codes import ReasonCode
from app.schemas.scenario import CityGraph, GraphZone, RoadEdge


class RoutePlanningError(ValueError):
    def __init__(self, message: str, reason_code: ReasonCode):
        super().__init__(message)
        self.reason_code = reason_code


@dataclass(frozen=True)
class PlannedPath:
    origin_zone_id: str
    destination_zone_id: str
    edge_ids: tuple[str, ...]
    distance_km: float
    travel_time_min: float
    mean_grade_percent: float
    grade_profile_percent: tuple[float, ...]


def effective_speed_kmh(
    edge: RoadEdge, traffic_factor: float, min_speed_factor: float = 0.0
) -> float:
    """Traffic-adjusted edge speed: 1/(1 + sensitivity*(factor-1)) scaling."""
    divisor = 1.0 + edge.traffic_sensitivity * max(0.0, traffic_factor - 1.0)
    speed = edge.free_flow_speed_kmh / max(divisor, 1e-6)
    floor = edge.free_flow_speed_kmh * min_speed_factor
    return max(speed, floor)


class GraphModel:
    """Adjacency view over a validated CityGraph."""

    def __init__(self, graph: CityGraph, min_speed_factor: float = 0.0) -> None:
        self.graph = graph
        self.min_speed_factor = min_speed_factor
        self.zones: dict[str, GraphZone] = {z.zone_id: z for z in graph.zones}
        self.edges: dict[str, RoadEdge] = {e.edge_id: e for e in graph.edges}
        self._adjacency: dict[str, list[tuple[RoadEdge, str]]] = {}
        self._build_adjacency()
        self.validate()

    def _build_adjacency(self) -> None:
        for edge in self.graph.edges:
            self._adjacency.setdefault(edge.from_node, []).append((edge, edge.to_node))
            if edge.bidirectional:
                self._adjacency.setdefault(edge.to_node, []).append((edge, edge.from_node))

    def validate(self) -> None:
        for edge in self.graph.edges:
            for endpoint in (edge.from_node, edge.to_node):
                if endpoint not in self.zones:
                    raise RoutePlanningError(
                        f"edge {edge.edge_id} references unknown node {endpoint}",
                        ReasonCode.ROUTE_EDGE_MISSING,
                    )
        reachable = self.reachable_zones(self.zones[next(iter(self.zones))].zone_id)
        if len(reachable) != len(self.zones):
            missing = sorted(set(self.zones) - reachable)
            raise RoutePlanningError(
                f"graph is disconnected; unreachable zones: {missing}",
                ReasonCode.ROUTE_EDGE_MISSING,
            )

    def reachable_zones(self, start: str) -> set[str]:
        seen = {start}
        stack = [start]
        while stack:
            node = stack.pop()
            for _edge, nxt in self._adjacency.get(node, []):
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        return seen

    def zones_of_kind(self, kind: str) -> list[GraphZone]:
        return [z for z in self.graph.zones if z.kind == kind]

    def edge_time_min(self, edge: RoadEdge, traffic_factor: float) -> float:
        speed = effective_speed_kmh(edge, traffic_factor, self.min_speed_factor)
        return edge.distance_km / (speed / 60.0)

    def shortest_path(
        self, origin_zone_id: str, destination_zone_id: str, traffic_factor: float
    ) -> PlannedPath:
        if origin_zone_id not in self.zones:
            raise RoutePlanningError(
                f"unknown origin zone {origin_zone_id}", ReasonCode.ROUTE_ZONE_UNKNOWN
            )
        if destination_zone_id not in self.zones:
            raise RoutePlanningError(
                f"unknown destination zone {destination_zone_id}", ReasonCode.ROUTE_ZONE_UNKNOWN
            )
        if origin_zone_id == destination_zone_id:
            raise RoutePlanningError(
                "origin and destination are the same zone", ReasonCode.ROUTE_ZONE_UNKNOWN
            )
        dist: dict[str, float] = {origin_zone_id: 0.0}
        prev: dict[str, tuple[str, RoadEdge]] = {}
        queue: list[tuple[float, str]] = [(0.0, origin_zone_id)]
        visited: set[str] = set()
        while queue:
            cost, node = heapq.heappop(queue)
            if node in visited:
                continue
            visited.add(node)
            if node == destination_zone_id:
                break
            for edge, nxt in self._adjacency.get(node, []):
                step = self.edge_time_min(edge, traffic_factor)
                new_cost = cost + step
                if new_cost < dist.get(nxt, float("inf")):
                    dist[nxt] = new_cost
                    prev[nxt] = (node, edge)
                    heapq.heappush(queue, (new_cost, nxt))
        if destination_zone_id not in prev and destination_zone_id != origin_zone_id:
            raise RoutePlanningError(
                f"no path from {origin_zone_id} to {destination_zone_id}",
                ReasonCode.ROUTE_EDGE_MISSING,
            )
        edges: list[RoadEdge] = []
        node = destination_zone_id
        while node != origin_zone_id:
            parent, edge = prev[node]
            edges.append(edge)
            node = parent
        edges.reverse()
        return self.to_path(origin_zone_id, destination_zone_id, edges, traffic_factor)

    def to_path(
        self,
        origin_zone_id: str,
        destination_zone_id: str,
        edges: list[RoadEdge],
        traffic_factor: float,
    ) -> PlannedPath:
        if not edges:
            raise RoutePlanningError("path contains no edges", ReasonCode.ROUTE_EDGE_MISSING)
        known = {e.edge_id for e in edges if e.edge_id in self.edges}
        if len(known) != len(edges):
            raise RoutePlanningError(
                "path references unknown edges", ReasonCode.ROUTE_EDGE_MISSING
            )
        distance_km = sum(e.distance_km for e in edges)
        travel_min = sum(self.edge_time_min(e, traffic_factor) for e in edges)
        grades: list[float] = []
        for e in edges:
            grades.extend(e.grade_profile_percent)
        weight = sum(e.distance_km for e in edges)
        mean_grade = sum(
            e.mean_grade_percent * e.distance_km for e in edges
        ) / max(weight, 1e-9)
        return PlannedPath(
            origin_zone_id=origin_zone_id,
            destination_zone_id=destination_zone_id,
            edge_ids=tuple(e.edge_id for e in edges),
            distance_km=distance_km,
            travel_time_min=travel_min,
            mean_grade_percent=mean_grade,
            grade_profile_percent=tuple(grades),
        )
