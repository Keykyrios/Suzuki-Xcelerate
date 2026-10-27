"""Synthetic demand generation and short-horizon zone forecasting.

Counts follow a Negative Binomial over a configured diurnal/day-type shape;
destinations follow a gravity-style distance decay. Everything is seeded, and
every record carries its provenance label — none of this is observed demand.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from app.config.loader import ConfigBundle, get_config
from app.schemas.common import CabinMode, DayType, EvidenceLabel
from app.schemas.scenario import DemandContext, DemandRecord, GraphZone


def diurnal_multipliers(cfg: ConfigBundle | None = None) -> np.ndarray:
    c = cfg or get_config()
    return np.asarray(c.demo_v("demand.time_of_day_peaks"), dtype=float)


def day_type_weights(cfg: ConfigBundle | None = None) -> dict[DayType, float]:
    c = cfg or get_config()
    raw = dict(c.demo_v("demand.day_type_weights"))
    return {DayType(k): float(v) for k, v in raw.items()}


def service_weights(cfg: ConfigBundle | None = None) -> dict[CabinMode, float]:
    c = cfg or get_config()
    raw = dict(c.demo_v("demand.service_type_weights"))
    return {CabinMode(k.upper()): float(v) for k, v in raw.items()}


def day_type_multiplier(cfg: ConfigBundle | None = None) -> dict[DayType, float]:
    c = cfg or get_config()
    raw = dict(c.demo_v("demand.day_type_multipliers"))
    return {DayType(k): float(v) for k, v in raw.items()}


def negative_binomial_sample(rng: np.random.Generator, mean: float, dispersion: float) -> int:
    """Gamma-Poisson mixture: overdispersed count around `mean`."""
    if mean <= 0.0:
        return 0
    rate = rng.gamma(shape=dispersion, scale=mean / dispersion)
    return int(rng.poisson(rate))


def _weighted_choice(rng: np.random.Generator, keys: list, weights: list[float]):
    probs = np.asarray(weights, dtype=float)
    total = probs.sum()
    if total <= 0:
        probs = np.ones_like(probs) / len(probs)
    else:
        probs = probs / total
    return keys[int(rng.choice(len(keys), p=probs))]


def destination_probabilities(
    origin_id: str,
    zones: list[GraphZone],
    service_class: CabinMode,
    cfg: ConfigBundle | None = None,
) -> dict[str, float]:
    """Gravity-style destination preference conditioned on service class."""
    c = cfg or get_config()
    beta = float(c.demo_v("demand.od_beta_per_km"))
    airport_weight = float(c.demo_v("demand.airport_attraction_weight"))
    intercity_weight = float(c.demo_v("demand.intercity_attraction_weight"))
    origin = next((z for z in zones if z.zone_id == origin_id), None)
    if origin is None:
        raise KeyError(f"unknown origin zone {origin_id}")

    scores: dict[str, float] = {}
    for zone in zones:
        if zone.zone_id == origin_id:
            continue
        dist_km = math.hypot(zone.x_m - origin.x_m, zone.y_m - origin.y_m) / 1000.0
        attract = 1.0
        if zone.kind == "airport" and service_class == CabinMode.AIRPORT:
            attract = airport_weight
        elif zone.kind == "corridor" and service_class == CabinMode.INTERCITY_PREMIUM:
            attract = intercity_weight
        elif zone.kind == "airport" and service_class != CabinMode.AIRPORT:
            attract = 0.25
        scores[zone.zone_id] = attract * math.exp(-beta * dist_km)

    total = sum(scores.values())
    if total <= 0:
        n = len(scores)
        return {k: 1.0 / n for k in scores}
    return {k: v / total for k, v in scores.items()}


def generate_demand_records(
    zones: list[GraphZone],
    count: int,
    seed: int,
    cfg: ConfigBundle | None = None,
) -> list[DemandRecord]:
    """Seeded historical-style demand records for the starter graph."""
    c = cfg or get_config()
    rng = np.random.default_rng(seed)
    hours = list(range(24))
    hour_weights = diurnal_multipliers(c).tolist()
    dts = list(DayType)
    dt_weights = [day_type_weights(c)[d] for d in dts]
    classes = list(CabinMode)
    class_weights = [service_weights(c)[k] for k in classes]
    dispersion = float(c.demo_v("demand.negative_binomial_dispersion"))
    daily_mean = float(c.demo_v("demand.daily_requests_mean"))
    urban = [z for z in zones if z.kind == "urban"]
    corridor_gateways = [z for z in zones if z.kind == "corridor"]

    records: list[DemandRecord] = []
    for index in range(count):
        hour = int(_weighted_choice(rng, hours, hour_weights))
        day_type = _weighted_choice(rng, dts, dt_weights)
        service = _weighted_choice(rng, classes, class_weights)

        origin_pool = urban or zones
        origin = origin_pool[int(rng.integers(0, len(origin_pool)))]
        dest_probs = destination_probabilities(origin.zone_id, zones, service, c)

        # Intercity demand runs from the city to the corridor gateway.
        if service == CabinMode.INTERCITY_PREMIUM and corridor_gateways and urban:
            origin = urban[int(rng.integers(0, len(urban)))]
            gateway = corridor_gateways[0]
            others = [z.zone_id for z in zones if z.zone_id != origin.zone_id]
            if gateway.zone_id in others:
                spread = 0.15 / max(len(others) - 1, 1)
                dest_probs = {zid: spread for zid in others}
                dest_probs[gateway.zone_id] = 0.85
            else:
                dest_probs = destination_probabilities(origin.zone_id, zones, service, c)

        dest_keys = sorted(dest_probs)
        dest_weights = [dest_probs[k] for k in dest_keys]
        destination = str(_weighted_choice(rng, dest_keys, dest_weights))
        if destination == origin.zone_id:
            continue

        hour_share = hour_weights[hour] / max(sum(hour_weights), 1e-9)
        expected = daily_mean * hour_share * day_type_multiplier(c)[day_type]
        request_count = negative_binomial_sample(rng, expected, dispersion)
        records.append(
            DemandRecord(
                record_id=f"DR-{index:03d}",
                hour_of_day=hour,
                day_type=day_type,
                service_class=service,
                origin_zone_id=origin.zone_id,
                destination_zone_id=destination,
                request_count=request_count,
                source_status=EvidenceLabel.SYNTHETIC,
            )
        )
    return records


def demand_context(
    hour_of_day: int, day_type: DayType, multiplier: float, cfg: ConfigBundle | None = None
) -> DemandContext:
    c = cfg or get_config()
    return DemandContext(
        daily_requests_mean=int(c.demo_v("demand.daily_requests_mean")),
        dispersion=float(c.demo_v("demand.negative_binomial_dispersion")),
        demand_multiplier=multiplier,
        day_type=day_type,
        hour_of_day=hour_of_day,
        provenance=EvidenceLabel.SYNTHETIC,
    )


@dataclass(frozen=True)
class ZoneForecast:
    zone_id: str
    expected_requests: float
    confidence: float


def zone_demand_forecast(
    records: list[DemandRecord],
    zone_id: str,
    hour_of_day: int,
    lookahead_h: float,
    context: DemandContext,
    cfg: ConfigBundle | None = None,
) -> ZoneForecast:
    """Expected requests leaving `zone_id` over the next window, with confidence.

    Confidence reflects how much history the estimate rests on, floored by the
    configured minimum so the label never overstates a thin history.
    """
    c = cfg or get_config()
    smoothing = float(c.demo_v("demand.zone_attraction_smoothing"))
    floor = float(c.demo_v("demand.forecast_confidence_floor"))
    multipliers = diurnal_multipliers(c)
    day_scale = day_type_multiplier(c)[context.day_type]

    zone_records = [r for r in records if r.origin_zone_id == zone_id]
    history_mass = sum(r.request_count for r in zone_records) + smoothing
    total_mass = sum(r.request_count for r in records) + smoothing * max(len(records), 1)
    share = history_mass / max(total_mass, 1e-9)

    hours_span = [h % 24 for h in range(int(hour_of_day), int(hour_of_day + math.ceil(lookahead_h)) + 1)]
    hour_sum = float(sum(multipliers[h] for h in hours_span)) / max(len(hours_span), 1)
    daily = context.daily_requests_mean * day_scale * context.demand_multiplier
    expected = daily * share * hour_sum * (lookahead_h / 24.0) * 24.0

    support = min(1.0, len(zone_records) / max(lookahead_h, 1.0))
    confidence = max(floor, min(1.0, 0.4 * support + 0.6 * (1.0 - math.exp(-history_mass))))
    return ZoneForecast(zone_id=zone_id, expected_requests=round(expected, 3), confidence=round(confidence, 3))
