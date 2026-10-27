"""Cabin-state aggregation: multi-frame debouncing, confidence gating, and
manual fallback.

The tracker never commands a cabin-mode change; it proposes one when the
configured rules fire and always exposes which source (camera or manual) and
which confidence produced the current state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from app.config.loader import ConfigBundle, get_config
from app.schemas.common import CabinMode, CabinSource, LuggageClass
from app.schemas.reason_codes import ReasonCode
from app.schemas.vehicle import CabinState, SeatRegion
from app.simulation.model_config import mode_layouts
from app.vision.detector import CabinReading


@dataclass
class TrackerConfig:
    confidence_threshold: float
    min_frames: int
    debounce_window_s: float
    state_retention_s: float
    mode_proposal_confidence: float
    airport_large_luggage: int


def tracker_config(cfg: ConfigBundle | None = None) -> TrackerConfig:
    c = cfg or get_config()
    return TrackerConfig(
        confidence_threshold=float(c.op("vision.confidence_threshold")),
        min_frames=int(c.op("vision.min_frames_for_reading")),
        debounce_window_s=float(c.op("vision.debounce_window_s")),
        state_retention_s=float(c.op("vision.state_retention_s")),
        mode_proposal_confidence=float(c.op("vision.mode_proposal_confidence")),
        airport_large_luggage=int(c.op("vision.airport_proposal_large_luggage")),
    )


def luggage_mass_kg(class_counts: dict[LuggageClass, int], cfg: ConfigBundle | None = None) -> float:
    c = cfg or get_config()
    total = 0.0
    for luggage_class, count in class_counts.items():
        record = c.vehicle.parameter(f"luggage_classes.{luggage_class.value}")
        total += float(record.default_mass_kg) * count
    return round(total, 2)


def seat_regions_for(
    detections_by_center: list[tuple[float, float]], cfg: ConfigBundle | None = None
) -> list[SeatRegion]:
    """Map detection centers onto calibrated seat regions when a calibration
    exists; with no calibration the mapping stays empty instead of guessing."""
    c = cfg or get_config()
    regions = list(c.op("vision.seat_regions") or [])
    if not regions:
        return []
    occupied: list[SeatRegion] = []
    for region in regions:
        region_id, (x1, y1, x2, y2) = region[0], region[1]
        hit = any(x1 <= cx <= x2 and y1 <= cy <= y2 for cx, cy in detections_by_center)
        occupied.append(SeatRegion(region_id=str(region_id), occupied=hit))
    return occupied


def propose_mode(
    passenger_count: int,
    large_luggage: int,
    cfg: ConfigBundle | None = None,
) -> tuple[CabinMode | None, str | None]:
    """Mode proposal from passenger/luggage counts; a proposal, never a command."""
    c = cfg or get_config()
    thresholds = tracker_config(c)
    layouts = mode_layouts(c)
    if large_luggage >= thresholds.airport_large_luggage:
        layout = layouts[CabinMode.AIRPORT]
        if passenger_count <= layout.max_passengers and large_luggage <= layout.max_luggage_units:
            return (
                CabinMode.AIRPORT,
                f"{large_luggage} large bags proposed AIRPORT mode for {passenger_count} passengers",
            )
        return (
            None,
            f"{large_luggage} large bags exceed the AIRPORT layout for {passenger_count} passengers",
        )
    for mode in (CabinMode.COMMUTE, CabinMode.POOL):
        layout = layouts[mode]
        if (
            passenger_count <= layout.max_passengers
            and large_luggage <= layout.max_luggage_units
        ):
            return mode, None
    return None, None


@dataclass
class CabinStateTracker:
    """Aggregates detector readings into a debounced canonical CabinState."""

    vehicle_id: str
    cfg: ConfigBundle = field(default_factory=get_config)

    def __post_init__(self) -> None:
        self.settings = tracker_config(self.cfg)
        self._buffer: list[CabinReading] = []
        self._state: CabinState | None = None
        self._last_reading_time_s: float | None = None
        self._last_emitted_time_s: float | None = None
        self._manual_override = False

    # -- camera path ------------------------------------------------------
    def ingest(self, reading: CabinReading, now_s: float) -> tuple[CabinState | None, list[str]]:
        """Buffer one frame reading; emit a state when debounce and confidence allow."""
        self._buffer.append(reading)
        if len(self._buffer) > self.settings.min_frames:
            self._buffer = self._buffer[-self.settings.min_frames:]
        reasons: list[str] = []

        if reading.confidence < self.settings.confidence_threshold:
            reasons.append(ReasonCode.CABIN_LOW_CONFIDENCE_MANUAL_FALLBACK.value)
            self._buffer.clear()
            if self._state is not None and not self._manual_override:
                self._state = self._state.model_copy(update={"override_flag": False})
            return self._state, reasons

        if len(self._buffer) < self.settings.min_frames:
            return self._state, reasons

        if (
            self._last_emitted_time_s is not None
            and now_s - self._last_emitted_time_s < self.settings.debounce_window_s
        ):
            return self._state, reasons

        aggregated = self._aggregate(self._buffer)
        self._buffer.clear()
        self._last_reading_time_s = now_s
        if self._manual_override:
            reasons.append(ReasonCode.MODE_OVERRIDE_APPLIED.value)
        self._last_emitted_time_s = now_s
        self._manual_override = False
        self._state = self._build_state(aggregated, CabinSource.CAMERA, now_s, confidence=aggregated.confidence)
        reasons.append(ReasonCode.CABIN_STATE_DEBOUNCED.value)
        return self._state, reasons

    # -- manual path ------------------------------------------------------
    def set_manual(
        self,
        *,
        mode: CabinMode | None,
        passenger_count: int,
        luggage_class_counts: dict[LuggageClass, int],
        now_s: float,
    ) -> tuple[CabinState, list[str]]:
        """Operator input always wins over the camera until the next reading."""
        self._manual_override = True
        small = luggage_class_counts.get(LuggageClass.SMALL, 0)
        large = luggage_class_counts.get(LuggageClass.LARGE, 0)
        reading = CabinReading(
            person_count=passenger_count,
            small_luggage=small,
            large_luggage=large,
            confidence=1.0,
            detection_count=0,
        )
        state = self._build_state(reading, CabinSource.MANUAL, now_s, confidence=1.0, forced_mode=mode)
        state = state.model_copy(update={"override_flag": True})
        self._state = state
        self._last_reading_time_s = now_s
        self._last_emitted_time_s = now_s
        return state, [ReasonCode.MODE_OVERRIDE_APPLIED.value]

    # -- staleness --------------------------------------------------------
    def check_stale(self, now_s: float) -> tuple[CabinState | None, list[str]]:
        """After the retention window without evidence, request manual confirmation."""
        if self._last_reading_time_s is None:
            return self._state, []
        if now_s - self._last_reading_time_s > self.settings.state_retention_s:
            if self._state is not None and self._state.source == CabinSource.CAMERA:
                self._state = self._state.model_copy(update={"override_flag": False})
            return self._state, [ReasonCode.CABIN_STALE_MANUAL_CONFIRMATION.value]
        return self._state, []

    @property
    def state(self) -> CabinState | None:
        return self._state

    # -- internals --------------------------------------------------------
    def _aggregate(self, buffer: list[CabinReading]) -> CabinReading:
        persons = [r.person_count for r in buffer]
        # Median keeps a single spurious frame from swinging the count.
        import statistics

        return CabinReading(
            person_count=int(statistics.median(persons)),
            small_luggage=int(statistics.median([r.small_luggage for r in buffer])),
            large_luggage=int(statistics.median([r.large_luggage for r in buffer])),
            confidence=float(sum(r.confidence for r in buffer) / len(buffer)),
            detection_count=int(statistics.mean([r.detection_count for r in buffer])),
        )

    def _build_state(
        self,
        reading: CabinReading,
        source: CabinSource,
        now_s: float,
        confidence: float,
        forced_mode: CabinMode | None = None,
    ) -> CabinState:
        classes: dict[LuggageClass, int] = {}
        if reading.small_luggage:
            classes[LuggageClass.SMALL] = reading.small_luggage
        if reading.large_luggage:
            classes[LuggageClass.LARGE] = reading.large_luggage
        mass = luggage_mass_kg(classes, self.cfg)

        if forced_mode is not None:
            mode = forced_mode
            reason = "operator-selected mode"
        else:
            mode = CabinMode.POOL if self._state is None else self._state.mode
            proposed, reason = propose_mode(reading.person_count, reading.large_luggage, self.cfg)
            if proposed is not None and self._state is not None and self._state.source == CabinSource.MANUAL:
                mode = self._state.mode
            elif proposed is not None:
                mode = proposed

        return CabinState(
            vehicle_id=self.vehicle_id,
            mode=mode,
            passenger_count=reading.person_count,
            luggage_units=reading.luggage_units,
            luggage_class_counts=classes,
            luggage_mass_kg=mass,
            seat_occupancy=[],
            confidence=round(max(0.0, min(1.0, confidence)), 3),
            source=source,
            timestamp_utc=datetime.fromtimestamp(now_s, tz=timezone.utc),
            override_flag=source == CabinSource.MANUAL,
            proposed_mode=None if forced_mode is not None else propose_mode(
                reading.person_count, reading.large_luggage, self.cfg
            )[0],
            mode_proposal_reason=reason,
        )
