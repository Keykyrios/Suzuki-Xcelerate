"""Reserve handling: below-reserve trips are never reported as feasible."""

from __future__ import annotations

from app.schemas.common import ConstraintStatus
from app.schemas.reason_codes import ReasonCode
from app.simulation.route_energy import route_feasible


def test_low_soc_on_a_long_trip_is_flagged_below_reserve(run_route):
    result = run_route(departure_soc=0.18, distance_km=6.0, passenger_count=6)
    assert result.constraint_checks["arrival_soc_reserve"] == ConstraintStatus.FAIL
    assert any("below the configured" in w for w in result.warnings)
    assert route_feasible(result) is False
    assert result.reserve_margin < 0.0


def test_sufficient_soc_keeps_the_check_green(run_route):
    result = run_route(departure_soc=0.85, distance_km=3.0, passenger_count=2)
    assert result.constraint_checks["arrival_soc_reserve"] == ConstraintStatus.PASS
    assert ReasonCode.ARRIVAL_SOC_ABOVE_RESERVE.value not in result.warnings
    assert route_feasible(result) is True


def test_displayed_reserve_is_the_stricter_of_reserve_and_operating_band(run_route, cfg):
    result = run_route(departure_soc=0.7)
    expected = max(
        float(cfg.op("soc_policy.soc_reserve_min")),
        float(cfg.op("soc_policy.soc_operating_min")),
    )
    assert result.soc_reserve_min == expected


def test_capacity_violation_marks_every_check_failed(run_route):
    result = run_route(passenger_count=99)
    assert set(result.constraint_checks.values()) == {ConstraintStatus.FAIL}
    assert route_feasible(result) is False
    assert any("infeasible" in w for w in result.warnings)
    assert result.route_energy_kwh == 0.0  # no numbers invented for an infeasible request


def test_below_reserve_reason_code_shape_is_stable():
    assert ReasonCode.ARRIVAL_SOC_BELOW_RESERVE.value == "ARRIVAL_SOC_BELOW_RESERVE"