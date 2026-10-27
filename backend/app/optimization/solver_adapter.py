"""Replaceable MILP solver adapter.

The optimizer speaks only this interface: a dense objective, integrality flags,
variable bounds, and a list of linear row constraints. OR-Tools (SCIP/CBC) is
the primary backend and SciPy's HiGHS MILP the fallback; neither is required
to be available at runtime beyond what is installed. Solver failures surface
as truthful statuses, never as a fake optimum.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class LinearRow:
    """One constraint: lb <= coefficients @ x <= ub."""

    coefficients: np.ndarray
    lb: float
    ub: float


@dataclass(frozen=True)
class MilpProblem:
    objective: np.ndarray
    integrality: np.ndarray  # 1 = integer, 0 = continuous
    lower_bounds: np.ndarray
    upper_bounds: np.ndarray
    rows: list[LinearRow] = field(default_factory=list)


@dataclass(frozen=True)
class MilpSolution:
    x: np.ndarray | None
    status: str  # OPTIMAL | FEASIBLE | INFEASIBLE | TIMEOUT | ERROR
    solver_name: str
    solver_version: str
    runtime_s: float
    message: str = ""
    objective_value: float | None = None

    @property
    def solved(self) -> bool:
        return self.status in ("OPTIMAL", "FEASIBLE") and self.x is not None


def _solve_with_ortools(problem: MilpProblem, timeout_s: float, gap: float) -> MilpSolution:
    from ortools.linear_solver import pywraplp
    import ortools

    started = time.perf_counter()
    solver = pywraplp.Solver.CreateSolver("SCIP")
    if solver is None:
        solver = pywraplp.Solver.CreateSolver("CBC")
    if solver is None:
        raise RuntimeError("no OR-Tools MILP backend registered")

    infinity = solver.infinity()
    variables = []
    for index in range(len(problem.objective)):
        lb = float(problem.lower_bounds[index])
        ub = (
            float(problem.upper_bounds[index])
            if np.isfinite(problem.upper_bounds[index])
            else infinity
        )
        if problem.integrality[index]:
            var = solver.IntVar(lb, ub, f"x{index}")
        else:
            var = solver.NumVar(lb, ub, f"x{index}")
        variables.append(var)

    for row in problem.rows:
        nonzero = np.nonzero(row.coefficients)[0]
        expr = sum(float(row.coefficients[int(i)]) * variables[int(i)] for i in nonzero)
        lb = float(row.lb) if np.isfinite(row.lb) else -infinity
        ub = float(row.ub) if np.isfinite(row.ub) else infinity
        solver.Add(expr >= lb)
        solver.Add(expr <= ub)

    objective = solver.Objective()
    for index, coefficient in enumerate(problem.objective):
        if coefficient != 0.0:
            objective.SetCoefficient(variables[int(index)], float(coefficient))
    objective.SetMinimization()
    solver.SetTimeLimit(int(timeout_s * 1000))
    try:
        solver.SetSolverSpecificParametersAsString(f"limits/gap = {gap}")
    except Exception:  # noqa: BLE001 - the gap is advisory; some backends reject it
        pass

    code = solver.Solve()
    runtime = time.perf_counter() - started
    status_map = {
        pywraplp.Solver.OPTIMAL: "OPTIMAL",
        pywraplp.Solver.FEASIBLE: "FEASIBLE",
        pywraplp.Solver.INFEASIBLE: "INFEASIBLE",
        pywraplp.Solver.NOT_SOLVED: "TIMEOUT",
    }
    status = status_map.get(code, "ERROR")
    version = getattr(ortools, "__version__", "unknown")
    if status in ("OPTIMAL", "FEASIBLE"):
        x = np.array([var.solution_value() for var in variables])
        return MilpSolution(
            x=x,
            status=status,
            solver_name="ortools_scip",
            solver_version=version,
            runtime_s=runtime,
            objective_value=float(solver.Objective().Value()),
        )
    return MilpSolution(
        x=None,
        status=status,
        solver_name="ortools_scip",
        solver_version=version,
        runtime_s=runtime,
        message=f"OR-Tools returned solver code {code}",
    )


def _solve_with_scipy(problem: MilpProblem, timeout_s: float, gap: float) -> MilpSolution:
    from scipy.optimize import Bounds, LinearConstraint, milp
    import scipy

    started = time.perf_counter()
    rows = problem.rows
    if rows:
        matrix = np.vstack([row.coefficients for row in rows])
        lb = np.array([row.lb for row in rows])
        ub = np.array([row.ub for row in rows])
        constraint = LinearConstraint(matrix, lb, ub)
    else:
        constraint = None
    finite_upper = np.where(np.isfinite(problem.upper_bounds), problem.upper_bounds, np.inf)
    result = milp(
        c=np.asarray(problem.objective, dtype=float),
        integrality=np.asarray(problem.integrality, dtype=int),
        bounds=Bounds(np.asarray(problem.lower_bounds, dtype=float), finite_upper),
        constraints=constraint,
        options={"time_limit": timeout_s, "mip_rel_gap": gap},
    )
    runtime = time.perf_counter() - started
    version = getattr(scipy, "__version__", "unknown")
    if result.success and result.x is not None:
        status = "OPTIMAL" if getattr(result, "mip_gap", 1.0) == 0.0 else "FEASIBLE"
        return MilpSolution(
            x=np.asarray(result.x),
            status=status,
            solver_name="scipy_highs",
            solver_version=version,
            runtime_s=runtime,
            objective_value=float(result.fun),
        )
    if result.status == 2:
        status = "INFEASIBLE"
    elif result.status == 1:
        status = "TIMEOUT"
    else:
        status = "ERROR"
    return MilpSolution(
        x=None,
        status=status,
        solver_name="scipy_highs",
        solver_version=version,
        runtime_s=runtime,
        message=getattr(result, "message", ""),
    )


def solve_milp(
    problem: MilpProblem,
    *,
    timeout_s: float = 15.0,
    mip_gap: float = 0.001,
    preferred: str = "ortools_milp",
    allow_fallback: bool = True,
) -> MilpSolution:
    """Solve with the preferred adapter, falling back only when configured."""
    backends = {
        "ortools_milp": _solve_with_ortools,
        "scipy_milp": _solve_with_scipy,
    }
    if preferred not in backends:
        raise ValueError(f"unknown solver adapter '{preferred}'; known: {sorted(backends)}")
    order = [preferred]
    if allow_fallback:
        order.extend(name for name in backends if name != preferred)

    last: MilpSolution | None = None
    for name in order:
        try:
            solution = backends[name](problem, timeout_s, mip_gap)
        except Exception as exc:  # noqa: BLE001 - adapter must degrade truthfully
            last = MilpSolution(
                x=None,
                status="ERROR",
                solver_name=name,
                solver_version="unknown",
                runtime_s=0.0,
                message=str(exc),
            )
            continue
        if solution.status in ("OPTIMAL", "FEASIBLE", "INFEASIBLE"):
            return solution
        last = solution
    assert last is not None
    return last
