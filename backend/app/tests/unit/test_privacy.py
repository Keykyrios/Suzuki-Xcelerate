"""Privacy: no raw-frame persistence, no identity code, honest disclaimers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.vision.camera import CameraUnavailable, LocalCameraSource, open_frame_source
from app.vision.detector import CabinDetector
from app.vision.maintenance_vision import TRAINED_LABELS, MaintenanceCueClassifier

ROOT = Path(__file__).resolve().parents[4]
VISION_DIR = ROOT / "backend" / "app" / "vision"


def snapshot_tree(root: Path) -> set[str]:
    return {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}


def test_frame_retention_policy_is_in_memory_only(cfg):
    assert cfg.op("vision.frame_retention_policy") == "in_memory_only_discard_after_inference"


def test_no_frame_is_written_during_inference(cfg):
    detector = CabinDetector(cfg)
    data_root = ROOT / "data"
    before = snapshot_tree(data_root)
    frame = np.random.default_rng(1).integers(0, 255, size=(360, 480, 3), dtype=np.uint8)
    detector.read_frame(frame)
    detector.read_frame(np.zeros((360, 480, 3), dtype=np.uint8))
    after = snapshot_tree(data_root)
    assert after == before


def test_vision_sources_have_no_identity_code():
    forbidden = ("face_recognition", "facenet", "reidentification", "re-identification", "person_reid")
    for path in VISION_DIR.glob("*.py"):
        text = path.read_text(encoding="utf-8").lower()
        for token in forbidden:
            assert token not in text, f"{path.name} contains {token}"


def test_detector_labels_are_object_classes_only(cfg, monkeypatch):
    parameter = cfg.policy._parameters["vision.coco_category_map"]
    original = parameter.value
    try:
        parameter.value = {"person": "person"}
        detector = CabinDetector(cfg)
        assert "face" not in " ".join(detector.category_map.keys()).lower()
        assert set(detector.category_map.values()) <= {"person", "small", "large"}
    finally:
        parameter.value = original


def test_camera_source_closes_and_absent_hardware_reports_itself(cfg, tmp_path):
    source = open_frame_source(cfg)
    frame = source.read()
    assert frame is None or isinstance(frame, np.ndarray)
    source.close()
    # A device that does not exist reports itself instead of crashing.
    with pytest.raises(CameraUnavailable) as excinfo:
        LocalCameraSource(device_index=254)
    assert "did not open" in str(excinfo.value)


def test_missing_frame_directory_is_reported_as_camera_unavailable(cfg, monkeypatch, tmp_path):
    parameter = cfg.policy._parameters["vision.prerecorded_frames_path"]
    original = parameter.value
    try:
        parameter.value = str((tmp_path / "missing_frames").as_posix())
        with pytest.raises(CameraUnavailable) as excinfo:
            open_frame_source(cfg)
        assert "not found" in str(excinfo.value)
    finally:
        parameter.value = original


def test_maintenance_cue_never_claims_roadworthiness(cfg):
    assert "roadworthy" not in TRAINED_LABELS
    assert set(TRAINED_LABELS) <= {"no_obvious_issue_detected", "inspect"}
    assert "roadworthiness" in cfg.op("vision.maintenance_disclaimer").lower() or "not a formal" in str(
        cfg.op("vision.maintenance_disclaimer")
    ).lower()


def test_maintenance_classifier_reports_a_disclaimer(cfg):
    classifier = MaintenanceCueClassifier(cfg)
    frame = np.zeros((224, 224, 3), dtype=np.uint8)
    cue = classifier.classify(frame)
    assert cue.label in {*TRAINED_LABELS, "uncertain"}
    assert 0.0 <= cue.confidence <= 1.0
    assert cue.disclaimer == str(cfg.op("vision.maintenance_disclaimer"))


def test_open_frame_source_uses_the_configured_source(cfg):
    source = open_frame_source(cfg)
    try:
        assert cfg.op("vision.camera_source") in {"prerecorded", "camera"}
        assert hasattr(source, "read") and hasattr(source, "close")
    finally:
        source.close()