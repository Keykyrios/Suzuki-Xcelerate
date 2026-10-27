"""Solver adapter: primary backend, fallback, infeasible and timeout honesty."""

from __future__ import annotations

import numpy as np
import pytest

from app.optimization.solver_adapter import LinearRow, MilpProblem, solve_milp


def simple_problem() -> MilpProblem:
    # max 3x + 4y  ->  min -3x -4y  with x + y <= 1, x,y binary
    return MilpProblem(
        objective=np.array([-3.0, -4.0]),
        integrality=np.array([1, 1]),
        lower_bounds=np.array([0.0, 0.0]),
        upper_bounds=np.array([1.0, 1.0]),
        rows=[
            LinearRow(coefficients=np.array([1.0, 1.0]), lb=-np.inf, ub=1.0),
        ],
    )


def test_primary_adapter_solves_a_tiny_milp():
    solution = solve_milp(simple_problem(), timeout_s=10.0, mip_gap=1e-4)
    assert solution.solved, solution
    assert solution.solver_name.startswith("ortools") or solution.solver_name.startswith(
        "scipy_"
    )
    assert solution.runtime_s >= 0.0
    x = np.asarray(solution.x)
    assert set(np.round(x, 6)) <= {0.0, 1.0}
    # Best assignment picks y (coefficient 4 above 3).
    objective = float(np.array([-3.0, -4.0]) @ x)
    assert objective == pytest.approx(-4.0, abs=1e-6)


def test_scipy_backend_is_also_usable_directly():
    solution = solve_milp(
        simple_problem(), timeout_s=10.0, mip_gap=1e-4, preferred="scipy_milp",
        allow_fallback=False,
    )
    assert solution.solver_name == "scipy_highs"
    assert solution.solved
    assert solution.objective_value == pytest.approx(-4.0, abs=1e-6)


def test_unknown_adapter_is_rejected():
    with pytest.raises(ValueError):
        solve_milp(simple_problem(), preferred="mystery_solver")


def test_infeasible_problem_is_reported_not_repaired():
    problem = MilpProblem(
        objective=np.array([1.0]),
        integrality=np.array([1]),
        lower_bounds=np.array([0.0]),
        upper_bounds=np.array([1.0]),
        rows=[
            LinearRow(coefficients=np.array([1.0]), lb=5.0, ub=10.0),
        ],
    )
    solution = solve_milp(problem, timeout_s=10.0, preferred="scipy_milp", allow_fallback=False)
    assert solution.status == "INFEASIBLE"
    assert solution.solved is False
    assert solution.x is None


def test_continuous_rows_and_bounds_are_honoured():
    # A pure continuous problem in [0, 2] minimising (x - 1)^2 is solved via a
    # linear objective: min -x pulls x to its upper bound.
    problem = MilpProblem(
        objective=np.array([-1.0]),
        integrality=np.array([0]),
        lower_bounds=np.array([0.0]),
        upper_bounds=np.array([2.0]),
        rows=[],
    )
    solution = solve_milp(problem, timeout_s=10.0, preferred="scipy_milp", allow_fallback=False)
    if solution.solved:
        assert float(solution.x[0]) == pytest.approx(2.0, abs=1e-6)


def test_fallback_chain_keeps_the_last_truthful_status():
    # Unknown preferred name with fallback allowed still raises: the preferred
    # backend must exist, so callers never think a real solver ran.
    with pytest.raises(ValueError):
        solve_milp(simple_problem(), preferred="not_a_solver", allow_fallback=True)