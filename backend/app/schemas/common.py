"""Enumerations and shared primitives for the canonical state model."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class CabinMode(str, Enum):
    POOL = "POOL"
    COMMUTE = "COMMUTE"
    AIRPORT = "AIRPORT"
    INTERCITY_PREMIUM = "INTERCITY_PREMIUM"


class PowertrainType(str, Enum):
    NEXUS_E = "NEXUS_E"
    NEXUS_X = "NEXUS_X"


class VehicleAvailability(str, Enum):
    AVAILABLE = "AVAILABLE"
    ASSIGNED = "ASSIGNED"
    CHARGING = "CHARGING"
    OFFLINE = "OFFLINE"
    MAINTENANCE = "MAINTENANCE"


class ChargerStatus(str, Enum):
    ONLINE = "ONLINE"
    OFFLINE = "OFFLINE"
    MAINTENANCE = "MAINTENANCE"


class RequestStatus(str, Enum):
    PENDING = "PENDING"
    ASSIGNED = "ASSIGNED"
    SERVED = "SERVED"
    UNSERVED = "UNSERVED"
    CANCELLED = "CANCELLED"


class CabinSource(str, Enum):
    CAMERA = "camera"
    MANUAL = "manual"
    SCENARIO = "scenario"


class LuggageClass(str, Enum):
    SMALL = "small"
    LARGE = "large"


class ConfidenceLabel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class ConstraintStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    SKIPPED = "SKIPPED"


class SolverStatus(str, Enum):
    OPTIMAL = "OPTIMAL"
    FEASIBLE = "FEASIBLE"
    NO_FEASIBLE_PLAN = "NO_FEASIBLE_PLAN"
    TIMEOUT = "TIMEOUT"
    ERROR = "ERROR"
    BASELINE_FALLBACK = "BASELINE_FALLBACK"


class DayType(str, Enum):
    WEEKDAY = "weekday"
    SATURDAY = "saturday"
    SUNDAY = "sunday"


class EvidenceLabel(str, Enum):
    """Alias kept close to the state model for payload ergonomics."""

    OBSERVED_PUBLIC = "OBSERVED_PUBLIC"
    MEASURED_LOCAL = "MEASURED_LOCAL"
    SYNTHETIC = "SYNTHETIC"
    SCENARIO_ASSUMPTION = "SCENARIO_ASSUMPTION"
    DESIGN_TARGET = "DESIGN_TARGET"
    MODEL_ESTIMATE = "MODEL_ESTIMATE"
    RECONSTRUCTED = "RECONSTRUCTED"
    UNCALIBRATED = "UNCALIBRATED"


class StrictModel(BaseModel):
    """Base model: unknown fields are rejected everywhere state is validated."""

    model_config = {"extra": "forbid"}
