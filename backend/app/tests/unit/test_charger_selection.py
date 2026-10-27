"""Charger selection: queue-adjusted cost can prefer the farther charger."""

from __future__ import annotations

from datetime import datetime, timezone

from app.schemas.charger import ChargerState
from app.schemas.common import ChargerStatus
from app.simulation.charging import ChargingEconomics, evaluate_charger, total_cost_inr

ECONOMICS = ChargingEconomics(
    queue_cost_inr_per_min=20.0,
    detour_cost_inr_per_km=10.0,
    lost_productive_inr_per_min=5.0,
    energy_inr_per_kwh=9.0,
    dc_surcharge_inr_per_kwh=0.0,
    wear_inr_per_kwh=1.0,
)


def make_charger(charger_id: str, **overrides) -> ChargerState:
    payload = {
        "charger_id": charger_id,
        "node_id": f"N-{charger_id}",
        "status": ChargerStatus.ONLINE,
        "plug_count": 1,
        "plugs_in_use": 1,
        "rated_power_kw": 50.0,
        "site_limit_kw": 100.0,
        "tariff_inr_per_kwh": 10.0,
        "connectors": ["CCS"],
        "queue_length": 0,
        "queue_estimate_min": 0.0,
        "last_updated_utc": datetime(2026, 1, 1, tzinfo=timezone.utc),
    }
    payload.update(overrides)
    return ChargerState(**payload)


def test_queue_adjusted_cost_can_flip_a_closest_charger_choice(bev_battery):
    near_busy = make_charger("CH-NEAR", queue_length=6, queue_estimate_min=8.0)
    far_clear = make_charger("CH-FAR")
    near_est = evaluate_charger(
        charger=near_busy,
        battery=bev_battery,
        economics=ECONOMICS,
        soc_now=0.5,
        target_soc=0.85,
        detour_km=1.0,
        plug_count_in_use=0,
        hour_of_day=10,
        tariff_bands=(12.0, 8.0),
        tariff_windows=(8, 20, 22, 6),
    )
    far_est = evaluate_charger(
        charger=far_clear,
        battery=bev_battery,
        economics=ECONOMICS,
        soc_now=0.5,
        target_soc=0.85,
        detour_km=6.0,
        plug_count_in_use=0,
        hour_of_day=10,
        tariff_bands=(12.0, 8.0),
        tariff_windows=(8, 20, 22, 6),
    )
    assert total_cost_inr(far_est.cost) < total_cost_inr(near_est.cost)
