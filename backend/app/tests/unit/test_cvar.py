"""CVaR risk mechanics: loss convention, tail statistics, objective construction."""

from __future__ import annotations

import numpy as np
import pytest

from app.optimization.risk import (
    CvarObjective,
    LOSS_DEFINITION,
    build_risk_metrics,
    empirical_cvar,
    evaluate_cvar,
    risk_config,
    tail_probability,
)
from app.schemas.reason_codes import ReasonCode


def test_loss_definition_is_stable():
    assert LOSS_DEFINITION == "loss_equals_negative_profit"
    config = risk_config()
    assert config.loss_definition == LOSS_DEFINITION
    assert 0.0 < config.alpha < 1.0
    assert config.lambda_weight >= 0.0
    assert config.scenario_batch_size >= 1


def test_cvar_of_a_flat_sample_is_the_sample_value():
    losses = np.full(50, 7.5)
    assert empirical_cvar(losses, 0.95) == pytest.approx(7.5)
    assert tail_probability(losses, 0.95) == pytest.approx(1.0)


def test_cvar_dominates_the_mean_for_a_tailed_sample():
    losses = np.array([1.0] * 95 + [40.0] * 5)
    cvar = empirical_cvar(losses, 0.95)
    tail_count = max(1, int(np.ceil((1.0 - 0.95) * losses.size)))
    expected = float(np.sort(losses)[::-1][:tail_count].mean())
    assert cvar == pytest.approx(expected)
    assert cvar >= losses.mean()


def test_cvar_is_monotonic_in_alpha_for_a_fixed_sample():
    losses = np.arange(1, 101, dtype=float)
    low_alpha = empirical_cvar(losses, 0.90)
    high_alpha = empirical_cvar(losses, 0.99)
    assert high_alpha >= low_alpha >= losses.mean()


def test_invalid_alpha_and_empty_samples_are_rejected():
    with pytest.raises(ValueError):
        empirical_cvar(np.array([]), 0.95)
    with pytest.raises(ValueError):
        empirical_cvar(np.ones(10), 0.0)
    with pytest.raises(ValueError):
        empirical_cvar(np.ones(10), 1.5)


def test_tail_probability_tracks_the_alpha_quantile():
    losses = np.arange(1, 101, dtype=float)
    share = tail_probability(losses, 0.95)
    assert 0.0 <= share <= 1.0
    assert share == pytest.approx((losses >= np.quantile(losses, 0.95)).mean())


def test_evaluate_cvar_scales_the_tail_with_lambda():
    losses = np.array([10.0, 12.0, 8.0, 45.0, 9.0])
    penalised, cvar = evaluate_cvar(losses, 0.8, 0.25)
    assert cvar == pytest.approx(empirical_cvar(losses, 0.8))
    assert penalised == pytest.approx(0.25 * cvar)


def test_objective_layout_and_weights():
    losses = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    objective = CvarObjective.build(
        losses,
        decision_count=2,
        constant_losses=np.zeros(3),
        alpha=0.6,
        lambda_weight=0.5,
    )
    assert objective.objective.shape == (2 + 1 + 3,)
    assert objective.eta_index == 2
    assert objective.u_start == 3
    np.testing.assert_allclose(objective.objective[:2], losses.mean(axis=0))
    assert objective.objective[2] == pytest.approx(0.5)
    assert objective.objective[3] == pytest.approx(0.5 / (0.4 * 3))


def test_cvar_rows_encode_the_auxiliary_constraints():
    losses = np.array([[1.0, 2.0], [3.0, 4.0]])
    objective = CvarObjective.build(
        losses, decision_count=2, constant_losses=np.zeros(2), alpha=0.9, lambda_weight=1.0
    )
    rows = objective.cvar_rows(losses)
    assert len(rows) == 2
    for index, (coefficients, ub) in enumerate(rows):
        assert coefficients.shape == (2 + 1 + 2,)
        assert ub == 0.0
        np.testing.assert_allclose(coefficients[:2], losses[index])
        assert coefficients[objective.eta_index] == -1.0
        assert coefficients[objective.u_start + index] == -1.0


def test_wrong_shapes_are_rejected():
    with pytest.raises(ValueError):
        CvarObjective.build(
            np.ones(4), decision_count=2, constant_losses=np.zeros(4), alpha=0.9, lambda_weight=1.0
        )
    with pytest.raises(ValueError):
        CvarObjective.build(
            np.ones((4, 2)), decision_count=2, constant_losses=np.ones(3),
            alpha=0.9, lambda_weight=1.0,
        )


def test_risk_metrics_report_disabled_state_honestly():
    config = risk_config()
    metrics = build_risk_metrics(np.array([]), config)
    assert metrics.enabled is False
    assert metrics.cvar_loss_inr == 0.0
    assert metrics.loss_definition == LOSS_DEFINITION


def test_risk_metrics_include_count_and_tail_probability():
    config = risk_config()
    losses = np.array([5.0, 6.0, 7.0, 30.0, 8.0, 9.0, 11.0, 12.0, 13.0, 25.0])
    metrics = build_risk_metrics(losses, config)
    assert metrics.enabled == config.enabled
    assert metrics.scenario_count == len(losses)
    assert 0.0 <= metrics.tail_probability <= 1.0
    assert metrics.expected_profit_inr == pytest.approx(-losses.mean(), abs=0.05)


def test_risk_reason_codes_exist():
    assert ReasonCode.CVAR_RISK_ACTIVE
    assert ReasonCode.CVAR_DISABLED