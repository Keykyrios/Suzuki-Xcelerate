"""Local detector for person counts and luggage classes.

A COCO-pretrained PyTorch detector runs fully on-device from weights stored in
the repository; the weights path, hash and class mapping come from
configuration. Frames become detections and are then dropped - nothing about
an image is kept.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.config.loader import ConfigBundle, get_config

PERSON_LABEL = "person"


@dataclass(frozen=True)
class Detection:
    label: str  # person | small | large (luggage classes)
    score: float
    bbox_xyxy: tuple[float, float, float, float]


@dataclass(frozen=True)
class CabinReading:
    """One frame's worth of derived cabin quantities."""

    person_count: int
    small_luggage: int
    large_luggage: int
    confidence: float
    detection_count: int

    @property
    def luggage_units(self) -> int:
        return self.small_luggage + self.large_luggage


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class CabinDetector:
    """Wraps the local detector model; safe to keep loaded across requests."""

    def __init__(self, cfg: ConfigBundle | None = None, verify_hash: bool = True) -> None:
        from ultralytics import YOLO

        c = cfg or get_config()
        self.cfg = c
        self.weights_path = Path(str(c.op("vision.model_weights_path")))
        if not self.weights_path.is_file():
            raise FileNotFoundError(
                f"detector weights not found at {self.weights_path}; "
                "they must be present locally before startup"
            )
        expected = str(c.op("vision.model_weights_sha256"))
        actual = sha256_of(self.weights_path)
        if verify_hash and actual != expected:
            raise ValueError(
                f"detector weights hash mismatch: expected {expected[:16]}..., got {actual[:16]}..."
            )
        self.weights_sha256 = actual
        self.arch = str(c.op("vision.model_arch"))
        self.input_size = int(c.op("vision.detection_input_size_px"))
        self.score_threshold = float(c.op("vision.detection_score_threshold"))
        self.empty_frame_confidence = float(c.op("vision.empty_frame_confidence"))
        raw_map = dict(c.op("vision.coco_category_map"))
        self.category_map: dict[str, str] = {k: str(v) for k, v in raw_map.items()}
        self._model = YOLO(str(self.weights_path))
        self._model.to("cpu")

    @property
    def model_version(self) -> str:
        return f"{self.arch}:{self.weights_sha256[:12]}"

    def detect(self, frame_rgb: np.ndarray) -> list[Detection]:
        """Run the detector on one RGB frame; the frame itself is never stored."""
        if frame_rgb.ndim != 3 or frame_rgb.shape[2] != 3:
            raise ValueError(f"expected an HxWx3 RGB frame, got shape {frame_rgb.shape}")
        results = self._model.predict(
            frame_rgb,
            imgsz=self.input_size,
            conf=self.score_threshold,
            device="cpu",
            verbose=False,
        )
        detections: list[Detection] = []
        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue
            xyxy = boxes.xyxy.cpu().numpy()
            confidences = boxes.conf.cpu().numpy()
            class_ids = boxes.cls.cpu().numpy().astype(int)
            names = result.names
            for box, confidence, class_id in zip(xyxy, confidences, class_ids):
                coco_name = str(names[int(class_id)])
                label = self.category_map.get(coco_name)
                if label is None:
                    continue
                detections.append(
                    Detection(
                        label=label,
                        score=float(confidence),
                        bbox_xyxy=tuple(float(v) for v in box),
                    )
                )
        return detections

    def read_frame(self, frame_rgb: np.ndarray) -> CabinReading:
        """Aggregate one frame into cabin quantities with an aggregate confidence."""
        return self.aggregate_detections(self.detect(frame_rgb))

    def aggregate_detections(self, detections: list[Detection]) -> CabinReading:
        persons = [d for d in detections if d.label == PERSON_LABEL]
        small = [d for d in detections if d.label == "small"]
        large = [d for d in detections if d.label == "large"]
        if detections:
            confidence = float(np.mean([d.score for d in detections]))
        else:
            # An empty frame cannot distinguish an empty cabin from missed
            # detections, so it carries the configured neutral confidence.
            confidence = self.empty_frame_confidence
        return CabinReading(
            person_count=len(persons),
            small_luggage=len(small),
            large_luggage=len(large),
            confidence=confidence,
            detection_count=len(detections),
        )
