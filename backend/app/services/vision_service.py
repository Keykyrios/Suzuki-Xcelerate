"""Cabin vision orchestration: camera frames in, canonical cabin state out.

Frames are consumed in memory and discarded after inference. The service owns
one aggregation tracker per vehicle so camera readings, manual overrides, and
the camera-off fallback all flow through the same debounced path.
"""

from __future__ import annotations

import time

from app.config.loader import ConfigBundle, get_config
from app.schemas.common import CabinMode, CabinSource, LuggageClass
from app.schemas.vehicle import CabinState
from app.services.state_store import StateStore, get_state_store
from app.vision.cabin_state import CabinStateTracker
from app.vision.camera import FrameSource, open_frame_source
from app.vision.detector import CabinDetector


def get_vision_service(
    cfg: ConfigBundle | None = None, store: StateStore | None = None
) -> "VisionService":
    return VisionService(cfg or get_config(), store or get_state_store())


class VisionService:
    """Runs the local detector over live or sample frames and publishes the
    aggregated cabin state into the canonical store."""

    def __init__(self, cfg: ConfigBundle, store: StateStore):
        self.cfg = cfg
        self.store = store
        self.camera_enabled = True
        self._detector: CabinDetector | None = None
        self._detector_error: str | None = None
        self._source: FrameSource | None = None
        self._trackers: dict[str, CabinStateTracker] = {}
        self._last_summaries: dict[str, dict] = {}

    # -- lazy local resources --------------------------------------------
    def _detector_or_error(self) -> CabinDetector | None:
        if self._detector is None and self._detector_error is None:
            try:
                self._detector = CabinDetector(self.cfg)
            except (FileNotFoundError, ValueError) as exc:
                self._detector_error = str(exc)
        return self._detector

    def _frame_source(self) -> FrameSource:
        if self._source is None:
            self._source = open_frame_source(self.cfg)
        return self._source

    def _tracker(self, vehicle_id: str) -> CabinStateTracker:
        if vehicle_id not in self._trackers:
            self._trackers[vehicle_id] = CabinStateTracker(
                vehicle_id=vehicle_id, cfg=self.cfg
            )
        return self._trackers[vehicle_id]

    # -- operator surfaces -------------------------------------------------
    def capture(self, vehicle_id: str, frame_count: int | None = None) -> dict:
        """Ingest a burst of frames and emit the debounced cabin state."""
        self.store.require_scenario().vehicle(vehicle_id)
        if not self.camera_enabled:
            raise RuntimeError("camera is off; last cabin state retained for operator input")
        detector = self._detector_or_error()
        if detector is None:
            raise RuntimeError(f"detector unavailable: {self._detector_error}")
        source = self._frame_source()
        count = frame_count or int(self.cfg.op("vision.min_frames_for_reading"))
        tracker = self._tracker(vehicle_id)
        emitted: CabinState | None = None
        reasons: list[str] = []
        readings = 0
        for _ in range(max(1, count)):
            frame = source.read()
            if frame is None:
                break
            reading = detector.read_frame(frame)
            readings += 1
            emitted, reasons = tracker.ingest(reading, now_s=time.monotonic())
        summary = {
            "vehicle_id": vehicle_id,
            "frames_ingested": readings,
            "model_version": detector.model_version,
            "camera_enabled": self.camera_enabled,
            "reason_codes": reasons,
        }
        if emitted is None:
            summary["state_emitted"] = False
            summary["note"] = (
                "reading below confidence threshold or debounce window; "
                "previous cabin state retained"
            )
        else:
            summary["state_emitted"] = True
            summary["state"] = emitted.model_dump(mode="json")
            self._publish(vehicle_id, emitted)
        self._last_summaries[vehicle_id] = summary
        return summary

    def process_frame(self, vehicle_id: str, frame_jpeg: "bytes") -> dict:
        """Run one browser-supplied JPEG frame through detection and aggregation.

        The frame lives only for the duration of inference; nothing is stored.
        """
        import base64

        import cv2
        import numpy as np

        self.store.require_scenario().vehicle(vehicle_id)
        if not self.camera_enabled:
            raise RuntimeError("camera is off; last cabin state retained for operator input")
        detector = self._detector_or_error()
        if detector is None:
            raise RuntimeError(f"detector unavailable: {self._detector_error}")
        encoded = frame_jpeg.split(b",", 1)[-1] if frame_jpeg.startswith(b"data:") else frame_jpeg
        buffer = np.frombuffer(base64.b64decode(encoded), dtype=np.uint8)
        bgr = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("frame could not be decoded as JPEG")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        detections = detector.detect(rgb)
        reading = detector.aggregate_detections(detections)
        state, reasons = self._tracker(vehicle_id).ingest(reading, now_s=time.monotonic())
        payload = {
            "vehicle_id": vehicle_id,
            "reading": {
                "person_count": reading.person_count,
                "small_luggage": reading.small_luggage,
                "large_luggage": reading.large_luggage,
                "confidence": round(reading.confidence, 3),
                "detection_count": reading.detection_count,
            },
            "detections": [
                {
                    "label": d.label,
                    "score": round(d.score, 3),
                    "bbox_xyxy": [round(v, 1) for v in d.bbox_xyxy],
                }
                for d in detections
            ],
            "state_emitted": state is not None and "CABIN_STATE_DEBOUNCED" in reasons,
            "reason_codes": reasons,
            "model_version": detector.model_version,
        }
        if state is not None:
            payload["state"] = state.model_dump(mode="json")
            if payload["state_emitted"]:
                self._publish(vehicle_id, state)
        return payload

    def manual_override(
        self,
        vehicle_id: str,
        *,
        mode: CabinMode,
        passenger_count: int,
        small_luggage: int,
        large_luggage: int,
    ) -> dict:
        """Operator-entered cabin state; wins over the camera until next reading."""
        self.store.require_scenario().vehicle(vehicle_id)
        state, reasons = self._tracker(vehicle_id).set_manual(
            mode=mode,
            passenger_count=passenger_count,
            luggage_class_counts={
                LuggageClass.SMALL: small_luggage,
                LuggageClass.LARGE: large_luggage,
            },
            now_s=time.monotonic(),
        )
        self._publish(vehicle_id, state)
        return {
            "vehicle_id": vehicle_id,
            "state": state.model_dump(mode="json"),
            "reason_codes": reasons,
        }

    def set_camera_enabled(self, enabled: bool) -> dict:
        """Camera off keeps the last cabin state for the configured retention."""
        self.camera_enabled = bool(enabled)
        return {
            "camera_enabled": self.camera_enabled,
            "retention_s": float(self.cfg.op("vision.state_retention_s")),
        }

    def status(self) -> dict:
        snapshot = self.store.snapshot()
        detector = self._detector_or_error()
        return {
            "camera_enabled": self.camera_enabled,
            "camera_source": str(self.cfg.op("vision.camera_source")),
            "detector_available": detector is not None,
            "detector_error": self._detector_error,
            "model_version": detector.model_version if detector else None,
            "confidence_threshold": float(self.cfg.op("vision.confidence_threshold")),
            "cabin_states": snapshot.cabin_states,
            "last_captures": dict(self._last_summaries),
        }

    # -- internals ----------------------------------------------------------
    def _publish(self, vehicle_id: str, state: CabinState) -> None:
        self.store.set_cabin_state(state)
        self.store.update_vehicle(
            vehicle_id,
            {
                "passenger_count": state.passenger_count,
                "luggage_units": state.luggage_units,
                "luggage_mass_kg": state.luggage_mass_kg,
                "cabin_mode": state.mode,
            },
        )
