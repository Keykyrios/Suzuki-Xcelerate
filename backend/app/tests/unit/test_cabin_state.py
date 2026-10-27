"""Cabin-state tracker: debounce, confidence gating, manual fallback, staleness."""

from __future__ import annotations

import pytest

from app.schemas.common import CabinMode, CabinSource, LuggageClass
from app.schemas.reason_codes import ReasonCode
from app.vision.cabin_state import (
    CabinStateTracker,
    luggage_mass_kg,
    propose_mode,
    seat_regions_for,
    tracker_config,
)
from app.vision.detector import CabinReading


def reading(persons=2, small=0, large=1, confidence=0.9, detections=3) -> CabinReading:
    return CabinReading(
        person_count=persons,
        small_luggage=small,
        large_luggage=large,
        confidence=confidence,
        detection_count=detections,
    )


@pytest.fixture
def tracker(cfg) -> CabinStateTracker:
    return CabinStateTracker(vehicle_id="V01", cfg=cfg)


def test_settings_come_from_configuration(cfg):
    settings = tracker_config(cfg)
    assert settings.confidence_threshold == pytest.approx(
        float(cfg.op("vision.confidence_threshold"))
    )
    assert settings.min_frames >= 1
    assert settings.debounce_window_s >= 0.0
    assert settings.state_retention_s > 0.0
    assert settings.airport_large_luggage >= 1


def test_state_is_withheld_until_min_frames_arrive(tracker, cfg):
    minimum = tracker_config(cfg).min_frames
    for index in range(minimum - 1):
        state, reasons = tracker.ingest(reading(), now_s=float(index))
        assert state is None
        assert reasons == []


def test_buffered_frames_emit_one_debounced_state(tracker, cfg):
    minimum = tracker_config(cfg).min_frames
    state = None
    reasons: list[str] = []
    for index in range(minimum):
        state, reasons = tracker.ingest(reading(persons=4), now_s=float(index))
    assert state is not None
    assert ReasonCode.CABIN_STATE_DEBOUNCED.value in reasons
    assert state.passenger_count == 4
    assert state.source == CabinSource.CAMERA
    assert 0.0 <= state.confidence <= 1.0


def test_debounce_suppresses_flickering_followups(tracker, cfg):
    settings = tracker_config(cfg)
    minimum = settings.min_frames
    for index in range(minimum):
        state, _ = tracker.ingest(reading(persons=2), now_s=float(index))
    assert state is not None
    for index in range(minimum):
        follow, _ = tracker.ingest(
            reading(persons=6), now_s=float(minimum + index) * 0.01
        )
    assert follow.passenger_count == state.passenger_count  # window too short

    after_window = settings.debounce_window_s + 1.0
    fresh = None
    for index in range(minimum):
        fresh, reasons = tracker.ingest(reading(persons=6), now_s=after_window + index)
    assert fresh is not None
    assert fresh.passenger_count == 6


def test_low_confidence_reads_request_manual_fallback(tracker, cfg, cfg_min_confidence=None):
    threshold = tracker_config(cfg).confidence_threshold
    emitted = None
    for index in range(tracker_settings_min_frames(cfg)):
        emitted, _ = tracker.ingest(
            reading(persons=3, confidence=threshold - 0.15), now_s=float(index)
        )
    assert emitted is None
    state, reasons = tracker.ingest(
        reading(persons=3, confidence=threshold - 0.15), now_s=100.0
    )
    assert ReasonCode.CABIN_LOW_CONFIDENCE_MANUAL_FALLBACK.value in reasons
    assert state is None or state.passenger_count >= 0


def tracker_settings_min_frames(cfg) -> int:
    return int(tracker_config(cfg).min_frames)


def test_median_aggregation_resists_a_spurious_frame(tracker, cfg):
    minimum = tracker_settings_min_frames(cfg)
    frames = [reading(persons=4), reading(persons=1), reading(persons=4)]
    state = None
    for index, frame in enumerate(frames[:minimum]):
        state, _ = tracker.ingest(frame, now_s=float(index))
    if state is None:  # min_frames may exceed the buffer; feed up to the window
        for index, frame in enumerate(frames):
            state, _ = tracker.ingest(frame, now_s=float(index))
    assert state is not None
    assert state.passenger_count == 4


def test_manual_override_wins_over_the_camera(tracker):
    state, reasons = tracker.set_manual(
        mode=CabinMode.AIRPORT,
        passenger_count=3,
        luggage_class_counts={LuggageClass.LARGE: 2},
        now_s=10.0,
    )
    assert state.mode == CabinMode.AIRPORT
    assert state.source == CabinSource.MANUAL
    assert state.override_flag is True
    assert ReasonCode.MODE_OVERRIDE_APPLIED.value in reasons
    assert tracker.state is state


def test_large_luggage_proposes_airport_without_commanding_it(tracker, cfg):
    settings = tracker_config(cfg)
    minimum = settings.min_frames
    state = None
    reasons: list[str] = []
    for index in range(minimum):
        state, reasons = tracker.ingest(
            reading(persons=3, large=settings.airport_large_luggage), now_s=float(index)
        )
    assert state is not None
    assert state.mode == CabinMode.AIRPORT
    assert state.proposed_mode == CabinMode.AIRPORT
    assert state.mode_proposal_reason
    assert state.source == CabinSource.CAMERA
    assert state.override_flag is False


def test_staleness_asks_for_manual_confirmation(tracker, cfg):
    settings = tracker_config(cfg)
    for index in range(settings.min_frames):
        tracker.ingest(reading(), now_s=float(index))
    _state, reasons = tracker.check_stale(settings.state_retention_s + 5.0)
    assert ReasonCode.CABIN_STALE_MANUAL_CONFIRMATION.value in reasons
    fresh_state, fresh_reasons = tracker.check_stale(settings.state_retention_s)
    assert fresh_reasons == []


def test_staleness_before_any_reading_is_silent(tracker):
    state, reasons = tracker.check_stale(12345.0)
    assert state is None
    assert reasons == []


@pytest.mark.parametrize(
    ("persons", "large", "expected"),
    [
        (0, 0, CabinMode.COMMUTE),
        (3, 0, CabinMode.COMMUTE),
        (3, 3, CabinMode.AIRPORT),
        (2, 1, CabinMode.COMMUTE),
    ],
)
def test_mode_proposal_rules(persons, large, expected, cfg):
    mode, reason = propose_mode(persons, large, cfg)
    assert mode == expected
    if mode is CabinMode.AIRPORT:
        assert reason


def test_mode_proposal_returns_none_when_the_layout_cannot_fit(cfg):
    settings = tracker_config(cfg)
    mode, reason = propose_mode(0, settings.airport_large_luggage, cfg)
    # Zero passengers with airport-level luggage still fits the AIRPORT layout,
    # but an impossible luggage count for the layout must explain itself.
    assert mode is not None or reason is not None
    oversized, explanation = propose_mode(0, 99, cfg)
    assert oversized is None
    assert explanation and "exceed" in explanation


def test_luggage_mass_uses_configured_class_masses(cfg):
    mass = luggage_mass_kg({LuggageClass.SMALL: 2, LuggageClass.LARGE: 1}, cfg)
    expected = 2 * float(cfg.vehicle.parameter("luggage_classes.small").default_mass_kg) + float(
        cfg.vehicle.parameter("luggage_classes.large").default_mass_kg
    )
    assert mass == pytest.approx(expected)


def test_seat_regions_stay_empty_without_a_calibration(cfg, monkeypatch):
    parameter = cfg.policy._parameters["vision.seat_regions"]
    original = parameter.value
    try:
        parameter.value = []
        assert seat_regions_for([(10.0, 10.0)], cfg) == []
    finally:
        parameter.value = original


def test_seat_regions_map_detections_onto_calibrated_boxes(cfg, monkeypatch):
    parameter = cfg.policy._parameters["vision.seat_regions"]
    original = parameter.value
    try:
        parameter.value = [["R1", [0, 0, 50, 50]], ["R2", [50, 0, 100, 50]]]
        regions = seat_regions_for([(25.0, 25.0)], cfg)
        assert [r.region_id for r in regions] == ["R1", "R2"]
        assert regions[0].occupied is True
        assert regions[1].occupied is False
        unoccupied = seat_regions_for([], cfg)
        assert [r.region_id for r in unoccupied] == ["R1", "R2"]
        assert all(r.occupied is False for r in unoccupied)
    finally:
        parameter.value = original


def test_tracker_state_is_never_claimed_as_measured(tracker):
    for index in range(tracker_settings_min_frames(__import__("app.config.loader", fromlist=["get_config"]).get_config())):
        state, _ = tracker.ingest(reading(), now_s=float(index))
    assert state is None or state.source in {CabinSource.CAMERA, CabinSource.MANUAL}