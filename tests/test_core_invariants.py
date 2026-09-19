"""Numerical-invariant guards on the core mcarma API, targeting the bug classes
that have actually bitten this project (see the file-based memory):

  * the Van Loan discrete-noise SIGN bug -- Qd must be PSD and reduce to the
    continuous integral GVG^T*dt at small dt (a sign flip makes Qd indefinite);
  * the stationary covariance must solve the continuous Lyapunov equation;
  * the Sigma soft-box FLOOR-height fix -- the floor center is 2*log(floor_mult*sd)
    with the corrected default floor_mult=1e-4 (the old 0.1 drove the (2,1) collapse);
  * the matrix PSD must be Hermitian positive-semidefinite with a positive diagonal;
  * AICc must use the small-sample correction and guard n_obs-k-1<=0.

Pure numpy/scipy, no optimizer, no JAX -- safe anywhere, fast.
Run: PYTHONUTF8=1 python tests/test_core_invariants.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mcarma.optimizer_utils import build_state_space          # noqa: E402
from mcarma.statespace import transition_and_noise, stationary_cov  # noqa: E402
from mcarma.simulate import mcarma_psd                        # noqa: E402
from mcarma.priors import sigma_softbox                       # noqa: E402
from mcarma.model_utils import compute_aicc, count_params     # noqa: E402


def _state_space(d=2, p=2, q=1, zeta=0.4, wn=0.08, rho=0.6):
    """Small coherent (p,q) state space with a 0.9^|i-j|-style Sigma."""
    if p == 1:
        ar = [[(wn,)] for _ in range(d)]
    else:
        ar = [[(2.0 * zeta * wn, wn ** 2)] for _ in range(d)]
    ma = [[(1.0 / (2.5 * wn),)] if q == 1 else [] for _ in range(d)]
    idx = np.arange(d)
    Sigma = rho ** np.abs(idx[:, None] - idx[None, :])
    F, G, H, V = build_state_space(ar, ma, Sigma, d, p, q)
    return F, G, H, V


def test_qd_lyapunov_psd_and_smalldt():
    """Discrete process noise Qd is PSD and -> GVG^T*dt as dt->0 (sign-bug guard)."""
    for p, q in [(1, 0), (2, 0), (2, 1)]:
        F, G, H, V = _state_space(p=p, q=q)
        # small-dt leading order: Qd/dt -> G V G^T
        dt = 1e-4
        Phi, Q = transition_and_noise(F, G, V, dt)
        assert np.allclose(Q, Q.T, atol=1e-12), (p, q, "Qd not symmetric")
        w = np.linalg.eigvalsh(Q)
        assert w.min() > -1e-10, (p, q, "Qd not PSD", w.min())
        GVG = G @ V @ G.T
        # first-order term; relative to the driving scale
        assert np.allclose(Q / dt, GVG, atol=1e-2 * (np.linalg.norm(GVG) + 1e-12)), (
            p, q, "Qd/dt does not reduce to GVG^T")
        # moderate dt: the defining identity Qd = P0 - Phi P0 Phi^T holds and stays PSD
        P0 = stationary_cov(F, G, V)
        Phi2, Q2 = transition_and_noise(F, G, V, 5.0)
        assert np.allclose(Q2, P0 - Phi2 @ P0 @ Phi2.T, atol=1e-8), (p, q)
        assert np.linalg.eigvalsh(Q2).min() > -1e-8, (p, q, "Qd@dt=5 not PSD")
        print(f"OK  Qd PSD + small-dt limit + Lyapunov identity  ({p},{q})")


def test_stationary_cov_solves_continuous_lyapunov():
    """stationary_cov P0 solves F P0 + P0 F^T + G V G^T = 0."""
    for p, q in [(1, 0), (2, 0), (2, 1)]:
        F, G, H, V = _state_space(p=p, q=q)
        P0 = stationary_cov(F, G, V)
        resid = F @ P0 + P0 @ F.T + G @ V @ G.T
        assert np.allclose(resid, 0.0, atol=1e-8), (p, q, np.abs(resid).max())
        assert np.linalg.eigvalsh(P0).min() > -1e-10, (p, q, "P0 not PSD")
        print(f"OK  stationary_cov solves continuous Lyapunov  ({p},{q})")


def test_sigma_softbox_floor_ceiling_and_default():
    """Floor/ceiling centers are 2*log(mult*sd_b); corrected default floor_mult=1e-4."""
    rng = np.random.default_rng(0)
    d = 3
    y = np.concatenate([rng.normal(0, s, 200) for s in (0.1, 0.2, 0.4)])
    band = np.concatenate([np.full(200, b) for b in range(d)])
    ceiling_mult = 5.0
    pk = sigma_softbox(y, band, d, lam=1.0, ceiling_mult=ceiling_mult)
    assert pk["chol_ridge_lambda"] == 1.0
    for b in range(d):
        sd_b = float(np.std(y[band == b]))
        assert np.isclose(pk["chol_ridge_center"][b], 2.0 * np.log(1e-4 * sd_b)), b
        assert np.isclose(pk["chol_ridge_center_hi"][b],
                          2.0 * np.log(ceiling_mult * sd_b)), b
    # lam<=0 disables the prior entirely
    assert sigma_softbox(y, band, d, lam=0.0, ceiling_mult=ceiling_mult) == {}
    # regression guard on the corrected default (0.1 was the (2,1)-collapse bug)
    import inspect
    assert inspect.signature(sigma_softbox).parameters["floor_mult"].default == 1e-4
    print("OK  sigma_softbox floor/ceiling centers + default floor_mult=1e-4")


def test_mcarma_psd_hermitian_psd():
    """P(omega) is Hermitian PSD with a real positive diagonal at every frequency."""
    F, G, H, V = _state_space(p=2, q=1)
    freqs = np.linspace(1e-3, 2.0, 40)
    P = np.asarray(mcarma_psd(F, G, H, V, freqs))
    assert P.shape[0] == len(freqs)
    for k in range(len(freqs)):
        Pk = P[k]
        assert np.allclose(Pk, Pk.conj().T, atol=1e-9), ("not Hermitian", k)
        w = np.linalg.eigvalsh(0.5 * (Pk + Pk.conj().T))
        assert w.min() > -1e-9, ("not PSD", k, w.min())
        assert np.all(np.real(np.diag(Pk)) > 0), ("nonpositive diagonal", k)
    print("OK  mcarma_psd Hermitian PSD, positive diagonal")


def test_compute_aicc_formula_and_guard():
    n, k, ll = 400, 12, -1230.10
    exp = -2 * ll + 2 * k + 2 * k * (k + 1) / (n - k - 1)
    assert np.isclose(compute_aicc(ll, k, n), exp)
    # small-sample guard: n - k - 1 <= 0 -> inf, never a negative correction
    assert compute_aicc(ll, k=n - 1, n_obs=n) == np.inf
    assert compute_aicc(ll, k=n, n_obs=n) == np.inf
    # count_params: AR (d*p) + MA (d*q) + Sigma (d(d+1)/2) + mu (d)
    assert count_params(5, 2, 1) == 5 * 2 + 5 * 1 + 5 * 6 // 2 + 5
    print("OK  compute_aicc small-sample correction + count_params")


if __name__ == "__main__":
    test_qd_lyapunov_psd_and_smalldt()
    test_stationary_cov_solves_continuous_lyapunov()
    test_sigma_softbox_floor_ceiling_and_default()
    test_mcarma_psd_hermitian_psd()
    test_compute_aicc_formula_and_guard()
    print("OK: core numerical invariants hold")
