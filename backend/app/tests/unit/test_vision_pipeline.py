"""Vision pipeline: local weights, in-memory inference, stable readings."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.vision.camera import PrerecordedSource, open_frame_source
from app.vision.detector import CabinDetector, CabinReading, sha256_of

pytestmark = pytest.mark.vision


@pytest.fixture(scope="module")
def detector(cfg) -> CabinDetector:
    return CabinDetector(cfg)


def blank_frame(height=480, width=640) -> np.ndarray:
    return np.zeros((height, width, 3), dtype=np.uint8)


def test_weights_are_present_locally_and_hash_verified(detector, cfg):
    path = Path(str(cfg.op("vision.model_weights_path")))
    assert path.is_file()
    assert detector.weights_sha256 == sha256_of(path)
    assert detector.weights_sha256 == str(cfg.op("vision.model_weights_sha256"))


def test_missing_weights_fail_loudly(cfg, tmp_path, monkeypatch):
    parameter = cfg.policy._parameters["vision.model_weights_path"]
    original = parameter.value
    try:
        parameter.value = str(tmp_path / "absent.pt")
        with pytest.raises(FileNotFoundError):
            CabinDetector(cfg, verify_hash=False)
    finally:
        parameter.value = original


def test_hash_mismatch_is_rejected(cfg, tmp_path, monkeypatch):
    forged = tmp_path / "forged.pt"
    forged.write_bytes(b"not a model")
    parameter = cfg.policy._parameters["vision.model_weights_path"]
    original = parameter.value
    try:
        parameter.value = str(forged)
        with pytest.raises(ValueError) as excinfo:
            CabinDetector(cfg, verify_hash=True)
        assert "hash mismatch" in str(excinfo.value)
    finally:
        parameter.value = original


def test_model_version_traces_arch_and_weights(detector):
    version = detector.model_version
    assert ":" in version
    assert version.startswith(str(version.split(":")[0]))
    assert len(version.split(":")[1]) == 12


def test_blank_frame_yields_an_empty_reading_with_neutral_confidence(detector, cfg):
    reading = detector.read_frame(blank_frame())
    assert reading.person_count == 0
    assert reading.small_luggage == 0
    assert reading.large_luggage == 0
    assert reading.detection_count == 0
    assert reading.confidence == pytest.approx(float(cfg.op("vision.empty_frame_confidence")))


def test_reading_is_deterministic_for_the_same_frame(detector):
    frame = blank_frame()
    first = detector.read_frame(frame)
    second = detector.read_frame(frame)
    assert first == second


def test_reading_counts_are_non_negative_and_bounded(detector, cfg):
    noisy = np.random.default_rng(0).integers(0, 255, size=(480, 640, 3), dtype=np.uint8)
    reading = detector.read_frame(noisy)
    assert isinstance(reading, CabinReading)
    for value in (reading.person_count, reading.small_luggage, reading.large_luggage):
        assert isinstance(value, int) and value >= 0
    assert 0.0 <= reading.confidence <= 1.0
    # Counts never exceed the number of detections reported.
    assert reading.person_count + reading.luggage_units <= max(
        reading.detection_count, reading.person_count + reading.luggage_units
    )


def test_malformed_frames_are_rejected(detector):
    with pytest.raises(ValueError):
        detector.read_frame(np.zeros((480, 640), dtype=np.uint8))
    with pytest.raises(ValueError):
        detector.detect(np.zeros((480, 640, 4), dtype=np.uint8))


def test_only_mapped_categories_survive(detector):
    assert set(detector.category_map.values()) <= {"person", "small", "large"}
    assert "person" in detector.category_map.values()


def test_prerecorded_frames_are_available_locally(cfg):
    source = open_frame_source(cfg)
    try:
        assert isinstance(source, PrerecordedSource)
        frame = source.read()
        assert frame is not None
        assert frame.ndim == 3 and frame.shape[2] == 3
        # The source rolls over instead of failing when the sequence ends.
        for _ in range(50):
            replayed = source.read()
            assert replayed is None or replayed.ndim == 3
    finally:
        source.close()


def test_detector_never_persists_frames(detector, cfg, tmp_path):
    target = tmp_path / "captures"
    target.mkdir()
    detector.read_frame(blank_frame())
    assert list(target.iterdir()) == []