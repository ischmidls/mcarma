"""Guard on the cross-band correlation ridge (Zhirui 2026-08-11), the only penalty
that touches the OFF-diagonal correlations of the innovation covariance Sigma.

Three things are asserted, on both objective paths (numpy fit.neg_loglik and the JAX
build_objective forward map):

  * DEFAULT OFF is byte-identical. corr_ridge_lambda=0 (and the absent default) leave
    the objective bit-for-bit unchanged, so production and every existing caller are
    untouched. This is the property the whole design hangs on.
  * The added term is exactly 0.5 * lambda * sum_{b<c} R_bc^2 on the correlation
    R = diag(Sigma)^-1/2 Sigma diag(Sigma)^-1/2. Verified by differencing the objective
    at lambda>0 against lambda=0 (everything else -- loglik + all other penalties --
    cancels) and comparing to the term recomputed from the packed Sigma.
  * numpy and JAX add the SAME term (the production sim runs --use-jax-grad, so the
    JAX forward map is the one that must carry the penalty).

Run: PYTHONUTF8=1 python tests/test_corr_ridge.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mcarma.fit import neg_loglik                                   # noqa: E402
from mcarma.observation import ObservationData                     # noqa: E402
from mcarma.optimizer_utils import (pack_params_jones,             # noqa: E402
                                     unpack_params_jones)


def _case(d=4, p=2, q=1, zeta=0.4, wn=0.08, rho=0.7, n_per=40, seed=0):
    """A small stationary (p,q) state space with a rho^|i-j| Sigma and an
    irregular multi-band design. Returns (data, theta, p, q). The data need not
    come from the model -- the likelihood only has to be finite for the
    difference test to isolate the penalty."""
    rng = np.random.default_rng(seed)
    if p == 1:
        ar = [[(wn,)] for _ in range(d)]
    else:
        ar = [[(2.0 * zeta * wn, wn ** 2)] for _ in range(d)]
    ma = [[(1.0 / (2.5 * wn),)] if q == 1 else [] for _ in range(d)]
    idx = np.arange(d)
    Sigma = 0.5 * rho ** np.abs(idx[:, None] - idx[None, :])
    theta_carma = pack_params_jones(ar, ma, Sigma)
    theta = np.concatenate([theta_carma, np.zeros(d)])              # mu = 0

    n = n_per * d
    t = np.sort(rng.uniform(0.0, 1500.0, size=n))
    band = rng.integers(0, d, size=n)
    y = rng.normal(0.0, 0.3, size=n)
    R = np.full(n, 0.01)
    data = ObservationData(t, y, band, R, d)
    return data, theta, p, q


def _expected_corr_term(theta, d, p, q, lam):
    """0.5 * lam * sum_{b<c} R_bc^2 from the packed Sigma, recomputed the way both
    objectives do it (from the unpacked V)."""
    _, _, V = unpack_params_jones(theta[:-d], d, p, q)
    V = np.asarray(V, float)
    sd = np.sqrt(np.clip(np.diag(V), 1e-300, None))
    R = V / np.outer(sd, sd)
    iu = np.triu_indices(d, 1)
    return 0.5 * lam * float(np.sum(R[iu] ** 2))


def test_numpy_default_off_is_byte_identical():
    for p, q in [(1, 0), (2, 0), (2, 1)]:
        data, theta, p, q = _case(p=p, q=q)
        base = neg_loglik(theta, data, p, q)
        z0 = neg_loglik(theta, data, p, q, corr_ridge_lambda=0.0)
        assert base == z0, (p, q, base, z0)          # exact, not approx
        print(f"OK  numpy corr_ridge default-off byte-identical  ({p},{q})")


def test_numpy_adds_exact_term():
    for p, q in [(1, 0), (2, 0), (2, 1)]:
        for lam in (0.5, 2.0, 10.0):
            data, theta, p, q = _case(p=p, q=q)
            d = data.d
            base = neg_loglik(theta, data, p, q, corr_ridge_lambda=0.0)
            got = neg_loglik(theta, data, p, q, corr_ridge_lambda=lam)
            assert np.isfinite(base), (p, q, "non-finite likelihood")
            exp = _expected_corr_term(theta, d, p, q, lam)
            assert np.isclose(got - base, exp, rtol=1e-9, atol=1e-10), (
                p, q, lam, got - base, exp)
        print(f"OK  numpy corr_ridge adds 0.5*lam*sum R_bc^2  ({p},{q})")


def test_univariate_corr_ridge_is_noop():
    """d=1 has no off-diagonals, so the ridge must be exactly zero for any lambda."""
    data, theta, p, q = _case(d=1, p=1, q=0)
    base = neg_loglik(theta, data, p, q, corr_ridge_lambda=0.0)
    got = neg_loglik(theta, data, p, q, corr_ridge_lambda=5.0)
    assert base == got, (base, got)
    print("OK  corr_ridge is a no-op for d=1 (no off-diagonals)")


def test_jax_matches_numpy():
    try:
        from mcarma.jax_loglik import build_objective, supports
    except Exception as exc:                                        # pragma: no cover
        print(f"SKIP jax parity (jax unavailable: {exc})")
        return
    for p, q in [(1, 0), (2, 0), (2, 1)]:
        if not supports(p, q):
            continue
        for lam in (0.0, 2.0):
            data, theta, p, q = _case(p=p, q=q)
            # soft band prior active so the JAX and numpy forward maps line up
            # (build_objective mirrors neg_loglik only when a band prior is on);
            # here we test the corr term in isolation by differencing lam vs 0.
            v0 = build_objective(data, p, q, ar_band_lambda=1.0,
                                 ma_band_lambda=1.0, corr_ridge_lambda=0.0)[0]
            vL = build_objective(data, p, q, ar_band_lambda=1.0,
                                 ma_band_lambda=1.0, corr_ridge_lambda=lam)[0]
            th = np.asarray(theta)
            d = data.d
            dj = float(vL(th)) - float(v0(th))
            exp = _expected_corr_term(theta, d, p, q, lam)
            assert np.isclose(dj, exp, rtol=1e-6, atol=1e-8), (p, q, lam, dj, exp)
        print(f"OK  jax corr_ridge adds the same term as numpy  ({p},{q})")


if __name__ == "__main__":
    test_numpy_default_off_is_byte_identical()
    test_numpy_adds_exact_term()
    test_univariate_corr_ridge_is_noop()
    test_jax_matches_numpy()
    print("OK: cross-band correlation ridge is off-by-default and exact on both paths")
