"""Restart ranking tiers: conditioning gate and the stationarity promotion.

The point under test is an inversion that is easy to get wrong. Because the
likelihood asymptotes as Sigma_bb -> 0, a fit sitting ON the boundary has a
vanishing gradient and therefore looks "converged". Ranking restarts on
stationarity alone would promote exactly those degenerate points above the
interior ones that are merely unfinished. The conditioning gate has to come
first, and these tests pin that ordering.
"""
import numpy as np
import pytest

from mcarma.fit import (SIGMA_COND_MAX, restart_quality_tier, sigma_condition)

D, P, Q = 3, 1, 0
N_CHOL = D * (D + 1) // 2


def _theta(logvars, offdiag=0.0):
    """Full packed theta (mu included) with the given Cholesky log-variances."""
    ar = np.full(P * D, -1.0)
    ma = np.full(Q * D, -1.0)
    chol = np.zeros(N_CHOL)
    rows, cols = np.tril_indices(D)
    diag_k = 0
    for k in range(N_CHOL):
        if rows[k] == cols[k]:
            chol[k] = logvars[diag_k]
            diag_k += 1
        else:
            chol[k] = offdiag
    mu = np.zeros(D)
    return np.concatenate([ar, ma, chol, mu])


INTERIOR = _theta([0.0, -0.5, 0.3])
COLLAPSED = _theta([0.0, -0.5, -120.0])


def test_sigma_condition_separates_the_two_populations():
    assert sigma_condition(INTERIOR, D, P, Q) < 1e3
    assert sigma_condition(COLLAPSED, D, P, Q) > SIGMA_COND_MAX


def test_sigma_condition_is_inf_on_garbage():
    assert sigma_condition(np.array([np.nan] * 12), D, P, Q) == np.inf
    assert sigma_condition(np.zeros(2), D, P, Q) == np.inf


def test_gates_off_is_the_historical_ranking():
    """Every scorable restart stays tier 1, so existing arms are reproducible."""
    for th in (INTERIOR, COLLAPSED):
        assert restart_quality_tier(th, -100.0, D, P, Q) == 1


def test_conditioning_gate_demotes_the_boundary():
    assert restart_quality_tier(INTERIOR, -100.0, D, P, Q,
                                cond_max=SIGMA_COND_MAX) == 2
    assert restart_quality_tier(COLLAPSED, -100.0, D, P, Q,
                                cond_max=SIGMA_COND_MAX) == 1


def test_flat_gradient_at_the_boundary_does_not_promote():
    """The inversion this whole design exists to prevent.

    A collapsed fit has ||grad|| ~ 0 and would be promoted to the top band by a
    naive stationarity rule. It must stay at tier 1.
    """
    flat = lambda th: 0.0
    assert restart_quality_tier(COLLAPSED, -100.0, D, P, Q,
                                cond_max=SIGMA_COND_MAX, grad=flat) == 1
    assert restart_quality_tier(INTERIOR, -100.0, D, P, Q,
                                cond_max=SIGMA_COND_MAX, grad=flat) == 3


def test_stationarity_needs_the_conditioning_gate_to_apply():
    """grad alone is inert: without cond_max nothing is promoted past tier 1."""
    assert restart_quality_tier(INTERIOR, -100.0, D, P, Q,
                                grad=lambda th: 0.0) == 1


def test_unfinished_interior_is_tier_2():
    steep = lambda th: 1e6
    assert restart_quality_tier(INTERIOR, -100.0, D, P, Q,
                                cond_max=SIGMA_COND_MAX, grad=steep) == 2


def test_gradient_tolerance_is_relative_to_the_score():
    """||g|| <= gtol_rel * max(1, |score|), the test finish_mle accepts on."""
    g = lambda th: 0.05
    # |score| = 1000 -> tol = 0.1, so 0.05 passes
    assert restart_quality_tier(INTERIOR, -1000.0, D, P, Q,
                                cond_max=SIGMA_COND_MAX, grad=g) == 3
    # |score| = 1 -> tol = 1e-4, so 0.05 fails
    assert restart_quality_tier(INTERIOR, -1.0, D, P, Q,
                                cond_max=SIGMA_COND_MAX, grad=g) == 2


def test_a_raising_gradient_does_not_crash_the_ranking():
    def boom(th):
        raise RuntimeError("no gradient here")
    assert restart_quality_tier(INTERIOR, -100.0, D, P, Q,
                                cond_max=SIGMA_COND_MAX, grad=boom) == 2


def test_interior_outranks_a_higher_likelihood_boundary_point():
    """The ordering fit() actually applies: (tier, score), tuple-compared."""
    flat = lambda th: 0.0
    good = (restart_quality_tier(INTERIOR, -900.0, D, P, Q,
                                 cond_max=SIGMA_COND_MAX, grad=flat), -900.0)
    bad = (restart_quality_tier(COLLAPSED, -100.0, D, P, Q,
                                cond_max=SIGMA_COND_MAX, grad=flat), -100.0)
    assert good > bad, "a boundary degeneracy won on likelihood alone"


def test_gates_off_lets_the_boundary_point_win_as_before():
    """Same pair, gates off: the old behaviour, kept deliberately."""
    good = (restart_quality_tier(INTERIOR, -900.0, D, P, Q), -900.0)
    bad = (restart_quality_tier(COLLAPSED, -100.0, D, P, Q), -100.0)
    assert bad > good
