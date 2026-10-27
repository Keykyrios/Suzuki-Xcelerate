"""Entropy-regularized origin-destination reconstruction.

Given origin and destination marginals plus a gravity prior over distances,
the iterative scaling solver finds the maximum-entropy flow matrix closest to
the prior that matches both marginals. Results are labelled RECONSTRUCTED:
they are inferred flows, never measured trips.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from app.config.loader import ConfigBundle, get_config
from app.schemas.common import EvidenceLabel
from app.schemas.scenario import GraphZone, ODFlow


@dataclass(frozen=True)
class ReconstructionResult:
    origin_ids: tuple[str, ...]
    destination_ids: tuple[str, ...]
    flows: np.ndarray
    iterations: int
    converged: bool
    marginal_residual: float
    beta_per_km: float
    constraints: dict[str, float] = field(default_factory=dict)
    provenance: EvidenceLabel = EvidenceLabel.RECONSTRUCTED

    def to_flows(self) -> list[ODFlow]:
        out: list[ODFlow] = []
        for i, origin in enumerate(self.origin_ids):
            for j, dest in enumerate(self.destination_ids):
                value = float(self.flows[i, j])
                if value > 1e-6 and origin != dest:
                    out.append(
                        ODFlow(
                            origin_zone_id=origin,
                            destination_zone_id=dest,
                            trips_per_day=round(value, 3),
                            provenance=EvidenceLabel.RECONSTRUCTED,
                        )
                    )
        return out


def gravity_prior(
    zones: list[GraphZone],
    beta_per_km: float,
    attraction: dict[str, float] | None = None,
) -> tuple[list[str], np.ndarray]:
    """Prior matrix: attraction_i * attraction_j * exp(-beta * distance)."""
    ids = [z.zone_id for z in zones]
    n = len(ids)
    prior = np.ones((n, n), dtype=float)
    for i, zi in enumerate(zones):
        for j, zj in enumerate(zones):
            if i == j:
                prior[i, j] = 0.0
                continue
            dist_km = math.hypot(zi.x_m - zj.x_m, zi.y_m - zj.y_m) / 1000.0
            a_i = (attraction or {}).get(zi.zone_id, 1.0)
            a_j = (attraction or {}).get(zj.zone_id, 1.0)
            prior[i, j] = a_i * a_j * math.exp(-beta_per_km * dist_km)
    return ids, prior


def reconstruct_od(
    origin_totals: np.ndarray,
    destination_totals: np.ndarray,
    zone_ids: list[str],
    prior: np.ndarray,
    *,
    cfg: ConfigBundle | None = None,
    max_iterations: int | None = None,
    tolerance: float | None = None,
    beta_per_km: float | None = None,
) -> ReconstructionResult:
    """Iteratively scale `prior` to match both marginals (Sinkhorn/IPF)."""
    c = cfg or get_config()
    iters = max_iterations if max_iterations is not None else int(
        c.demo_v("demand.od_reconstruction_max_iterations")
    )
    tol = tolerance if tolerance is not None else float(
        c.demo_v("demand.od_reconstruction_tolerance")
    )
    beta = beta_per_km if beta_per_km is not None else float(
        c.demo_v("demand.od_reconstruction_beta_per_km")
    )

    origins = np.asarray(origin_totals, dtype=float)
    dests = np.asarray(destination_totals, dtype=float)
    if origins.shape != dests.shape or origins.shape[0] != prior.shape[0]:
        raise ValueError(
            f"shape mismatch: origins {origins.shape}, dests {dests.shape}, prior {prior.shape}"
        )
    if np.any(origins < 0) or np.any(dests < 0):
        raise ValueError("marginal totals must be non-negative")
    if not np.isfinite(prior).all():
        raise ValueError("prior matrix contains non-finite values")

    flows = prior.copy()
    total = origins.sum()
    if total <= 0:
        raise ValueError("origin marginal must sum to a positive total")

    converged = False
    residual = float("inf")
    iteration = 0
    for iteration in range(1, iters + 1):
        row_sums = flows.sum(axis=1)
        flows *= np.where(row_sums > 0, origins / np.where(row_sums > 0, row_sums, 1.0), 1.0)[:, None]
        col_sums = flows.sum(axis=0)
        flows *= np.where(col_sums > 0, dests / np.where(col_sums > 0, col_sums, 1.0), 1.0)[None, :]
        row_err = np.abs(flows.sum(axis=1) - origins).max() if origins.size else 0.0
        col_err = np.abs(flows.sum(axis=0) - dests).max() if dests.size else 0.0
        residual = float(max(row_err, col_err))
        if residual < tol:
            converged = True
            break

    return ReconstructionResult(
        origin_ids=tuple(zone_ids),
        destination_ids=tuple(zone_ids),
        flows=flows,
        iterations=iteration,
        converged=converged,
        marginal_residual=residual,
        beta_per_km=beta,
        constraints={
            "origin_total_trips": round(float(origins.sum()), 3),
            "destination_total_trips": round(float(dests.sum()), 3),
            "zone_count": float(len(zone_ids)),
        },
    )


def reconstruct_from_records(
    zones: list[GraphZone],
    origin_totals: np.ndarray,
    destination_totals: np.ndarray,
    cfg: ConfigBundle | None = None,
) -> ReconstructionResult:
    """Convenience wrapper: gravity prior from config, then solve."""
    c = cfg or get_config()
    beta = float(c.demo_v("demand.od_reconstruction_beta_per_km"))
    zone_ids, prior = gravity_prior(zones, beta)
    return reconstruct_od(
        origin_totals, destination_totals, zone_ids, prior, cfg=c, beta_per_km=beta
    )
