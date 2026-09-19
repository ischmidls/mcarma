"""Guard on cross-band diagonal loading (Zhirui 2026-08-13), the alternative to the
correlation ridge. Unlike the ridge this is a MODEL regularizer, not an additive
penalty term: the innovation covariance V is replaced by V + lam*mean(diag(V))*I
INSIDE the likelihood (state-space P0 / transitions / Kalman), lifting the smallest
eigenvalue off the near-singular cross-band boundary. So the effect on the objective
is nonlinear (no closed-form added term), and the assertions differ from the ridge:

  * DEFAULT OFF is byte-identical. diag_load_lambda=0 (and the absent default) leave
    the objective bit-for-bit unchanged, so production and every existing caller are
    untouched. This is the property the whole design hangs on.
  * numpy and JAX apply the SAME loading. The loglik itself differs between the two
    forward maps (the JAX map mirrors neg_loglik only with a band prior on), so we
    difference lam>0 against lam=0 WITHIN each path -- the base loglik and all other
    penalties cancel, leaving only the loading effect -- and require the two deltas
    to agree. The production sim runs --use-jax-grad, so the JAX map is the one that
    must carry the regularizer.
  * NOT a no-op at d=1. With one band V is a scalar v and V + lam*(v/1)*I = (1+lam)*v,
    a genuine rescale, so unlike the correlation ridge diagonal loading DOES move the
    d=1 objective. (Pinned so a future "skip for d=1" shortcut can't slip in.)

Run: PYTHONUTF8=1 python tests/test_diag_load.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mcarma.fit import neg_loglik                                   # noqa: E402
from mcarma.observation import ObservationData                     # noqa: E402
from mcarma.optimizer_utils import pack_params_jones               # noqa: E402


def _case(d=4, p=2, q=1, zeta=0.4, wn=0.08, rho=0.7, n_per=40, seed=0):
    """A small stationary (p,q) state space with a rho^|i-j| Sigma and an
    irregular multi-band design. Returns (data, theta, p, q). The data need not
    come from the model -- the likelihood only has to be finite for the
    difference test to isolate the loading."""
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


def test_numpy_default_off_is_byte_identical():
    for p, q in [(1, 0), (2, 0), (2, 1)]:
        data, theta, p, q = _case(p=p, q=q)
        base = neg_loglik(theta, data, p, q)
        z0 = neg_loglik(theta, data, p, q, diag_load_lambda=0.0)
        assert base == z0, (p, q, base, z0)          # exact, not approx
        print(f"OK  numpy diag_load default-off byte-identical  ({p},{q})")


def test_numpy_loading_moves_the_objective():
    """Loading a near-singular cross-band Sigma must change the (finite) objective;
    a strong rho makes the smallest eigenvalue small so the effect is real."""
    for p, q in [(1, 0), (2, 0), (2, 1)]:
        data, theta, p, q = _case(p=p, q=q, rho=0.9)
        base = neg_loglik(theta, data, p, q, diag_load_lambda=0.0)
        got = neg_loglik(theta, data, p, q, diag_load_lambda=0.1)
        assert np.isfinite(base) and np.isfinite(got), (p, q, base, got)
        assert base != got, (p, q, "loading left objective unchanged", base)
        print(f"OK  numpy diag_load moves the objective  ({p},{q})  "
              f"d={got - base:+.4g}")


def test_univariate_is_not_a_noop():
    """d=1: V is a scalar v, V + lam*(v/1)*I = (1+lam)*v -- a real rescale, so
    UNLIKE the correlation ridge diagonal loading must NOT be a no-op at d=1."""
    data, theta, p, q = _case(d=1, p=1, q=0)
    base = neg_loglik(theta, data, p, q, diag_load_lambda=0.0)
    got = neg_loglik(theta, data, p, q, diag_load_lambda=0.5)
    assert np.isfinite(base) and np.isfinite(got), (base, got)
    assert base != got, (base, got, "d=1 loading was a no-op")
    print("OK  diag_load is a genuine rescale for d=1 (not a no-op)")


def test_jax_matches_numpy():
    try:
        from mcarma.jax_loglik import build_objective, supports
    except Exception as exc:                                        # pragma: no cover
        print(f"SKIP jax parity (jax unavailable: {exc})")
        return
    for p, q in [(1, 0), (2, 0), (2, 1)]:
        if not supports(p, q):
            continue
        for lam in (0.1, 0.3):
            data, theta, p, q = _case(p=p, q=q, rho=0.9)
            th = np.asarray(theta)
            d = data.d
            # numpy delta from loading (band priors on so the two forward maps line
            # up; the base loglik + band penalty cancel in the difference).
            n0 = neg_loglik(th, data, p, q, ar_band_lambda=1.0, ma_band_lambda=1.0,
                            diag_load_lambda=0.0)
            nL = neg_loglik(th, data, p, q, ar_band_lambda=1.0, ma_band_lambda=1.0,
                            diag_load_lambda=lam)
            # jax delta from the same loading on the JAX forward map.
            v0 = build_objective(data, p, q, ar_band_lambda=1.0,
                                 ma_band_lambda=1.0, diag_load_lambda=0.0)[0]
            vL = build_objective(data, p, q, ar_band_lambda=1.0,
                                 ma_band_lambda=1.0, diag_load_lambda=lam)[0]
            dj = float(vL(th)) - float(v0(th))
            dn = float(nL) - float(n0)
            assert np.isfinite(dn) and np.isfinite(dj), (p, q, lam, dn, dj)
            assert np.isclose(dj, dn, rtol=1e-6, atol=1e-8), (p, q, lam, dj, dn)
        print(f"OK  jax diag_load applies the same loading as numpy  ({p},{q})")


if __name__ == "__main__":
    test_numpy_default_off_is_byte_identical()
    test_numpy_loading_moves_the_objective()
    test_univariate_is_not_a_noop()
    test_jax_matches_numpy()
    print("OK: cross-band diagonal loading is off-by-default and matched on both paths")
