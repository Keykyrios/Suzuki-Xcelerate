"""Evidence-status taxonomy shared by configuration, API payloads, and exports."""

from __future__ import annotations

from enum import Enum


class EvidenceStatus(str, Enum):
    """Provenance/quality label attached to every configurable value and result."""

    OBSERVED_PUBLIC = "OBSERVED_PUBLIC"
    MEASURED_LOCAL = "MEASURED_LOCAL"
    SYNTHETIC = "SYNTHETIC"
    SCENARIO_ASSUMPTION = "SCENARIO_ASSUMPTION"
    DESIGN_TARGET = "DESIGN_TARGET"
    MODEL_ESTIMATE = "MODEL_ESTIMATE"
    RECONSTRUCTED = "RECONSTRUCTED"
    UNCALIBRATED = "UNCALIBRATED"


MODEL_OUTPUT_LABELS = {EvidenceStatus.MODEL_ESTIMATE, EvidenceStatus.UNCALIBRATED}
