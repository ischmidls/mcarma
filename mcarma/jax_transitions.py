"""
mcarma/jax_transitions.py
==========================
Batched discrete-time transition precomputation via JAX, with overflow guards.

Replaces the per-step _transition_cache loop in kalman.loglik.

Discretization
--------------
For ``dZ = F Z dt + G dB`` with ``Cov(dB)=V dt`` and stationary covariance
``P0`` (Lyapunov solution of ``F P0 + P0 Fᵀ + G V Gᵀ = 0``), the exact
discrete-time transition over a gap ``dt`` is

    Phi = exp(F·dt),   Q = P0 − Phi P0 Phiᵀ.

This Lyapunov form is used instead of the Van Loan augmented-matrix extraction.
The earlier Van Loan code built ``[[F, GVGᵀ],[0,-Fᵀ]]`` (a sign-flipped
convention) and so produced a wrong ``Q`` — off by ~2× even at moderate dt and
divergent at the ~100-unit seasonal gaps, where the anti-stable ``-Fᵀ`` block
makes ``expm`` overflow (Phi~1e39, Q~1e182, not even PSD). The Lyapunov form
needs only the bounded ``Phi`` and ``P0``, so it is stable for every dt and
matches the defining integral to floating-point precision.

Guards retained
---------------
- Non-finite fallback: if ``jax_expm(F·dt)`` is non-finite (only when F is
  near-unstable, i.e. an out-of-range parameter during optimisation), that dt
  falls back to ``Phi = 0, Q = P_stationary``.
- ``Q`` is symmetrised (``0.5*(Q+Q.T)``) and projected to PSD only if a tiny
  negative eigenvalue appears.
"""

import jax
import jax.numpy as jnp
from jax.scipy.linalg import expm as jax_expm
import numpy as np

jax.config.update("jax_enable_x64", True)


@jax.jit
def _lyap_batch(F_j, P0_j, dts):
    """
    Batched discrete-time transition via the exact Lyapunov form.

    Returns (Phis, Qs) each of shape (k, n, n), where

        Phi = exp(F*dt)
        Q   = P0 - Phi P0 Phi^T   ( = integral_0^dt e^{Fs} G V G^T e^{F^T s} ds ).

    This replaces the previous Van Loan extraction, which used a sign-flipped
    augmented matrix [[F, GVGT],[0,-F^T]] and so produced a wrong Qd (off by ~2x
    even at moderate dt, and divergent at large dt because the anti-stable -F^T
    block overflows expm). The Lyapunov form needs only the bounded Phi=exp(F*dt)
    and the stationary covariance P0, so it is stable for all dt.
    """
    def single(dt):
        Phi = jax_expm(F_j * dt)
        Q   = P0_j - Phi @ P0_j @ Phi.T
        Q   = 0.5 * (Q + Q.T)
        return Phi, Q

    return jax.vmap(single)(dts)


def precompute_transitions_jax(t_obs, F, G, Sigma,
                                lambda_max_real, P_stationary):
    """
    Precompute discrete-time transitions for all unique time gaps in t_obs.

    Replaces the ``_transition_cache`` loop in kalman.loglik.
    Returns ``Phi_arr``, ``Q_arr`` as lists of plain numpy arrays.

    Overflow guard
    --------------
    If ``jax_expm`` produces non-finite values for a given dt (which happens
    when ``G Σ Gᵀ * dt`` has entries ~1e8+, e.g. during optimisation with
    an inflated Cholesky init), that dt falls back to
        Phi = 0,  Q = P_stationary
    identical to the large-dt shortcut in ``transition_and_noise``.
    """
    n     = F.shape[0]
    n_obs = len(t_obs)
    eps   = 1e-10
    eye   = np.eye(n)
    zero_Q = np.zeros((n, n))

    dts = np.array([float(t_obs[k] - t_obs[k - 1]) for k in range(1, n_obs)])

    nonzero_mask = dts > 0.0
    unique_dts   = np.unique(np.round(dts[nonzero_mask], 4))

    # Large-dt shortcut: Phi → 0, Q → P_stationary
    if lambda_max_real < 0:
        large_mask   = np.exp(lambda_max_real * unique_dts) < eps
        small_unique = unique_dts[~large_mask]
    else:
        large_mask   = np.zeros(len(unique_dts), dtype=bool)
        small_unique = unique_dts

    results = {}

    # ── Batch expm over small unique dts ─────────────────────────────────────
    if len(small_unique) > 0:
        F_j  = jnp.array(F, dtype=jnp.float64)
        P0_j = jnp.array(P_stationary, dtype=jnp.float64)
        Phis_j, Qs_j = _lyap_batch(
            F_j, P0_j, jnp.array(small_unique, dtype=jnp.float64)
        )
        Phis_np = np.array(Phis_j)
        Qs_np   = np.array(Qs_j)

        for i, dt_key in enumerate(small_unique):
            Phi_i = Phis_np[i]
            Q_i   = Qs_np[i]

            # Overflow guard: fall back if expm produced non-finite output
            if not (np.isfinite(Phi_i).all() and np.isfinite(Q_i).all()):
                results[dt_key] = (np.zeros((n, n)), P_stationary.copy())
                continue

            Q_i = 0.5 * (Q_i + Q_i.T)   # symmetrise
            # Clamp tiny negative eigenvalues (mirror of project_pd intent)
            eigvals = np.linalg.eigvalsh(Q_i)
            if eigvals.min() < -1e-10:
                # Full projection only when needed — rare
                from mcarma.optimizer_utils import project_pd
                Q_i = project_pd(Q_i)

            results[dt_key] = (Phi_i, Q_i)

    # ── Large-dt fallback ─────────────────────────────────────────────────────
    for dt_key in unique_dts[large_mask]:
        results[dt_key] = (np.zeros((n, n)), P_stationary.copy())

    # ── Build per-observation arrays ──────────────────────────────────────────
    Phi_arr = [None] * n_obs
    Q_arr   = [None] * n_obs
    for k in range(1, n_obs):
        dt = float(t_obs[k] - t_obs[k - 1])
        if dt == 0.0:
            Phi_arr[k] = eye
            Q_arr[k]   = zero_Q
        else:
            # Look up with np.round (not builtin round): the cache keys are built
            # with np.round(dts, 4) at the top, and np.round vs builtin round
            # disagree on tie values (e.g. 5.93125 -> 5.9312 vs 5.9313), which
            # otherwise raises KeyError for cadences landing on that boundary.
            Phi_arr[k], Q_arr[k] = results[np.round(dt, 4)]

    return Phi_arr, Q_arr