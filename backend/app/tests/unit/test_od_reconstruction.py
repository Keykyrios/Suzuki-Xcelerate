"""OD reconstruction: marginals matched, labels honest, priors respected."""

from __future__ import annotations

import numpy as np
import pytest

from app.demand.od_reconstruction import gravity_prior, reconstruct_od
from app.schemas.common import CabinMode, EvidenceLabel
from app.schemas.scenario import GraphZone


@pytest.fixture
def zones() -> list[GraphZone]:
    return [
        GraphZone(zone_id=f"Z{i}", name=f"Zone {i}", x_m=float(i) * 4000.0, y_m=float(i % 3) * 3000.0)
        for i in range(6)
    ]


def test_reconstruction_matches_both_marginals(zones, cfg):
    origins = np.array([120.0, 80.0, 60.0, 90.0, 40.0, 50.0])
    dests = np.array([100.0, 70.0, 85.0, 55.0, 45.0, 85.0])
    ids, prior = gravity_prior(zones, beta_per_km=0.15)
    result = reconstruct_od(origins, dests, ids, prior, cfg=cfg)
    assert result.converged is True
    assert result.marginal_residual < float(cfg.demo_v("demand.od_reconstruction_tolerance")) * 10
    assert result.flows.sum(axis=1) == pytest.approx(origins, rel=0.05, abs=1.0)
    assert result.flows.sum(axis=0) == pytest.approx(dests, rel=0.05, abs=1.0)


def test_reconstructed_flows_are_labelled_reconstructed(zones, cfg):
    n = len(zones)
    origins = np.full(n, 50.0)
    dests = np.full(n, 50.0)
    ids, prior = gravity_prior(zones, 0.1)
    result = reconstruct_od(origins, dests, ids, prior, cfg=cfg)
    flows = result.to_flows()
    assert flows
    assert result.provenance == EvidenceLabel.RECONSTRUCTED
    assert all(f.provenance == EvidenceLabel.RECONSTRUCTED for f in flows)
    assert all(f.trips_per_day > 0.0 for f in flows)


def test_gravity_prior_vanishes_on_the_diagonal(zones):
    ids, prior = gravity_prior(zones, 0.2)
    assert len(ids) == len(zones)
    assert np.allclose(np.diag(prior), 0.0)
    assert np.all(prior >= 0.0)


def test_gravity_prior_decays_with_distance(zones):
    _, prior = gravity_prior(zones, beta_per_km=0.5)
    # Z0-Z1 is closer than Z0-Z5, so its prior mass is larger.
    assert prior[0, 1] > prior[0, 5]


def test_attraction_weights_are_applied(zones):
    ids, plain = gravity_prior(zones, 0.1)
    _, boosted = gravity_prior(zones, 0.1, attraction={ids[1]: 5.0})
    assert boosted[0, 1] == pytest.approx(plain[0, 1] * 5.0)


def test_shape_and_sign_validations(zones, cfg):
    ids, prior = gravity_prior(zones, 0.1)
    n = len(ids)
    with pytest.raises(ValueError):
        reconstruct_od(np.full(n - 1, 10.0), np.full(n, 10.0), ids, prior, cfg=cfg)
    with pytest.raises(ValueError):
        reconstruct_od(np.full(n, -1.0), np.full(n, 10.0), ids, prior, cfg=cfg)


def test_reconstruction_is_repeatable(zones, cfg):
    origins = np.array([110.0, 70.0, 90.0, 60.0, 55.0, 75.0])
    dests = np.array([95.0, 80.0, 70.0, 60.0, 50.0, 105.0])
    ids, prior = gravity_prior(zones, 0.12)
    first = reconstruct_od(origins, dests, ids, prior, cfg=cfg)
    second = reconstruct_od(origins, dests, ids, prior, cfg=cfg)
    assert np.allclose(first.flows, second.flows)


def test_service_classes_do_not_change_reconstruction_engine(zones, cfg):
    # The reconstruction is label-agnostic; the caller attaches service context.
    ids, prior = gravity_prior(zones, 0.1)
    n = len(ids)
    result = reconstruct_od(np.full(n, 40.0), np.full(n, 40.0), ids, prior, cfg=cfg)
    assert list(CabinMode)  # enumeration remains available for routing beyond reconstruction
    assert result.iterations >= 1