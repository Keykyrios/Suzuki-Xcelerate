"""Risk utilities: loss convention, empirical CVaR, and objective terms.

Convention (fixed across the implementation): loss = -profit, so a loss is a
positive plan cost and revenue shows up as a reduction. The risk-aware
objective minimises expected loss plus lambda * CVaR_alpha(loss), built with
the standard auxiliary-variable representation:

    u_s >= loss_s - eta
    u_s >= 0
    CVaR_alpha = eta + (1 / ((1 - alpha) * N)) * sum(u_s)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.config.loader import ConfigBundle, get_config
from app.schemas.decision import RiskMetrics

LOSS_DEFINITION = "loss_equals_negative_profit"


@dataclass(frozen=True)
class RiskConfig:
    enabled: bool
    alpha: float
    lambda_weight: float
    scenario_batch_size: int
    traffic_sigma: float
    demand_sigma: float

    @property
    def loss_definition(self) -> str:
        return LOSS_DEFINITION


def risk_config(cfg: ConfigBundle | None = None) -> RiskConfig:
    c = cfg or get_config()
    return RiskConfig(
        enabled=bool(c.opt("risk.enabled")),
        alpha=float(c.opt("risk.alpha")),
        lambda_weight=float(c.opt("risk.lambda")),
        scenario_batch_size=int(c.opt("risk.scenario_batch_size")),
        traffic_sigma=float(c.opt("risk.uncertainty_traffic_sigma")),
        demand_sigma=float(c.opt("risk.uncertainty_demand_sigma")),
    )


def empirical_cvar(losses: np.ndarray, alpha: float) -> float:
    """Expected loss of the worst (1 - alpha) tail of a discrete sample."""
    if losses.size == 0:
        raise ValueError("cannot compute CVaR of an empty loss sample")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha {alpha} outside (0, 1)")
    ordered = np.sort(losses)[::-1]
    tail_count = int(np.ceil((1.0 - alpha) * ordered.size))
    tail_count = max(1, min(tail_count, ordered.size))
    return float(ordered[:tail_count].mean())


def tail_probability(losses: np.ndarray, alpha: float, threshold: float | None = None) -> float:
    """Share of scenarios at or beyond the alpha-quantile loss."""
    if losses.size == 0:
        return 0.0
    if threshold is None:
        threshold = float(np.quantile(losses, alpha))
    return float((losses >= threshold).mean())


@dataclass(frozen=True)
class CvarObjective:
    """Coefficients of the auxiliary-variable CVaR objective.

    Variables are laid out as [decisions..., eta, u_0 .. u_{S-1}]; the
    objective vector returned here already blends the expected-loss term and
    the lambda-weighted CVaR term, so a caller can add it directly to any
    solver adapter.
    """

    objective: np.ndarray
    eta_index: int
    u_start: int
    scenario_count: int
    alpha: float
    lambda_weight: float

    @staticmethod
    def build(
        scenario_losses_by_decision: np.ndarray,
        *,
        decision_count: int,
        constant_losses: np.ndarray,
        alpha: float,
        lambda_weight: float,
    ) -> "CvarObjective":
        """Blend E[loss] + lambda * CVaR over decision-dependent scenario losses.

        `scenario_losses_by_decision` has shape (S, D): the loss contribution
        of each decision variable in each scenario. `constant_losses` has shape
        (S,): scenario losses that do not depend on decisions (e.g. penalties
        that apply regardless).
        """
        losses = np.asarray(scenario_losses_by_decision, dtype=float)
        constants = np.asarray(constant_losses, dtype=float)
        if losses.ndim != 2:
            raise ValueError(f"expected (S, D) matrix, got shape {losses.shape}")
        scenario_count, d = losses.shape
        if d != decision_count:
            raise ValueError(f"decision_count {decision_count} != matrix width {d}")
        if constants.shape != (scenario_count,):
            raise ValueError("constant_losses must have shape (S,)")

        eta_index = decision_count
        u_start = decision_count + 1
        total_vars = decision_count + 1 + scenario_count

        objective = np.zeros(total_vars)
        expected = losses.mean(axis=0)
        objective[:decision_count] = expected
        objective[eta_index] = lambda_weight
        objective[u_start:] = lambda_weight / ((1.0 - alpha) * scenario_count)
        # Constant expected part folds into nothing (does not affect argmin).
        _ = constants
        return CvarObjective(
            objective=objective,
            eta_index=eta_index,
            u_start=u_start,
            scenario_count=scenario_count,
            alpha=alpha,
            lambda_weight=lambda_weight,
        )

    def cvar_rows(self, scenario_losses_by_decision: np.ndarray) -> list[tuple[np.ndarray, float]]:
        """Rows enforcing u_s >= loss_s(decisions) - eta as <= 0 inequalities."""
        losses = np.asarray(scenario_losses_by_decision, dtype=float)
        scenario_count, decision_count = losses.shape
        rows = []
        for s in range(scenario_count):
            row = np.zeros(decision_count + 1 + scenario_count)
            row[:decision_count] = losses[s]
            row[self.eta_index] = -1.0
            row[self.u_start + s] = -1.0
            rows.append((row, 0.0))
        return rows


def evaluate_cvar(
    losses: np.ndarray, alpha: float, lambda_weight: float
) -> tuple[float, float]:
    """(lambda * CVaR, empirical CVaR) of a realized loss sample."""
    cvar = empirical_cvar(losses, alpha)
    return lambda_weight * cvar, cvar


def build_risk_metrics(
    scenario_losses_inr: np.ndarray,
    config: RiskConfig,
) -> RiskMetrics:
    """Package the computed tail risk for decision output."""
    losses = np.asarray(scenario_losses_inr, dtype=float)
    if not config.enabled or losses.size == 0:
        return RiskMetrics(
            enabled=False,
            alpha=config.alpha,
            lambda_weight=config.lambda_weight,
            loss_definition=LOSS_DEFINITION,
            expected_profit_inr=0.0,
            cvar_loss_inr=0.0,
            scenario_count=int(losses.size),
            tail_probability=0.0,
        )
    expected_loss = float(losses.mean())
    cvar = empirical_cvar(losses, config.alpha)
    return RiskMetrics(
        enabled=True,
        alpha=config.alpha,
        lambda_weight=config.lambda_weight,
        loss_definition=LOSS_DEFINITION,
        expected_profit_inr=round(-expected_loss, 2),
        cvar_loss_inr=round(cvar, 2),
        scenario_count=int(losses.size),
        tail_probability=round(tail_probability(losses, config.alpha), 4),
    )
