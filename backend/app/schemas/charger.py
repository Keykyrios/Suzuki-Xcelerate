"""Charger entities."""

from __future__ import annotations

from datetime import datetime

from pydantic import Field

from app.schemas.common import ChargerStatus, StrictModel


class ChargerState(StrictModel):
    charger_id: str
    node_id: str
    status: ChargerStatus
    plug_count: int = Field(ge=1, le=20)
    plugs_in_use: int = Field(ge=0, le=20)
    rated_power_kw: float = Field(gt=0.0)
    site_limit_kw: float = Field(gt=0.0)
    tariff_inr_per_kwh: float = Field(ge=0.0)
    connectors: list[str]
    queue_length: int = Field(ge=0, le=50)
    queue_estimate_min: float = Field(ge=0.0)
    last_updated_utc: datetime

    @property
    def plugs_available(self) -> int:
        return max(0, self.plug_count - self.plugs_in_use)

    @property
    def is_online(self) -> bool:
        return self.status == ChargerStatus.ONLINE
