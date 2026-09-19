"""
mcarma/statespace.py
====================
Exact discretization and stationary covariance computation for continuous-time
state-space systems.

Provides core functionality for MCARMA simulation and inference:
  1. transition_and_noise() - Exact discrete-time conversion: Phi = expm(F*dt)
     and process noise Qd via the stationary Lyapunov identity (NOT the Van Loan
     block-matrix extraction, which was sign-flipped/divergent -- see below).
  2. stationary_cov() - Stationary covariance via continuous Lyapunov equation

See docs/METHODS.Rmd sec 3 for the discretization math.

These functions are used by the simulation engine and Kalman filter to handle
irregularly spaced observations correctly.

Model
-----
Continuous-time linear stochastic system:
    dZ(t) = F Z(t) dt + G dB(t),   Cov(dB(t)) = V dt

Discrete-time equivalent at times t and t+Δt:
    Z_{k} = Φ(Δt) Z_{k-1} + w_k,   w_k ~ N(0, Q_d(Δt))

where:
    Φ(Δt) = exp(F·Δt)  (state transition matrix)
    Q_d(Δt) = ∫₀^{Δt} exp(F·s) G V Gᵀ exp(F·s)ᵀ ds  (discrete noise covariance)

Mathematical Details
--------------------
Φ is computed directly as Φ = exp(F·Δt). The integrated process-noise covariance
uses the stationary Lyapunov identity

    Q_d(Δt) = ∫₀^{Δt} e^{Fs} G V Gᵀ e^{Fᵀs} ds = P₀ − Φ P₀ Φᵀ,

where P₀ is the stationary covariance (continuous Lyapunov, below). This needs
only the bounded Φ and P₀, so it is stable for all Δt.

NOTE: the classic Van Loan (1978) block-matrix extraction
(M = [[-F, GVGᵀ], [0, Fᵀ]], Φ = E[:n,:n]ᵀ, Q = Φ·E[:n,n:]) is NOT used here — the
top-left −F block is *anti-stable* (its eigenvalues are −λᵢ, which have positive
real part because F is Hurwitz), so exp(M·Δt) grows like e^{|Re λ|·Δt} and
overflows past a few fast-mode time constants (~1e39 at a seasonal gap); an
earlier implementation here was additionally sign-flipped. The Lyapunov form above
replaces it and uses only the bounded, stable Φ. See docs/METHODS.Rmd §3.

Large-Δt Handling
-----------------
For stable systems (max real eigenvalue λ_max < 0), when exp(λ_max·Δt) < eps,
the transition matrix Φ effectively vanishes. In this regime:
    - Φ ≈ 0
    - Q_d ≈ stationary covariance P∞

This prevents numerical overflow and speeds up computation for large gaps.

Stationary Covariance
---------------------
For stable systems (all eigenvalues of F have negative real parts), the
stationary covariance P∞ satisfies the continuous Lyapunov equation:
    F P∞ + P∞ Fᵀ + G V Gᵀ = 0

This is solved using scipy.linalg.solve_continuous_lyapunov with a small
diagonal regularization to ensure positive definiteness.

Parameters
----------
F : (n, n) ndarray
    Drift/transition matrix (stable, all eigenvalues negative real part)
G : (n, d) ndarray
    Noise input matrix
V : (d, d) ndarray
    Latent driving covariance matrix (positive definite)
dt : float
    Time increment between observations (must be > 0)
lambda_max_real : float or None
    Pre-computed maximum real part of eigenvalues of F. If None, computed
    internally via np.linalg.eigvals.
P_stationary : (n, n) ndarray or None
    Pre-computed stationary covariance. If provided, used directly in
    large-Δt regime instead of recomputing.

Returns for transition_and_noise()
----------------------------------
Phi : (n, n) ndarray
    State transition matrix exp(F·dt)
Q   : (n, n) ndarray
    Discrete-time process noise covariance (symmetric positive definite)

Parameters for stationary_cov()
------------------------------
F : (n, n) ndarray
    Stable drift matrix
G : (n, d) ndarray
    Noise input matrix
V : (d, d) ndarray
    Latent driving covariance matrix
eps : float, default=1e-12
    Small diagonal regularization added to Lyapunov solution to ensure
    positive definiteness

Returns for stationary_cov()
----------------------------
P : (n, n) ndarray
    Stationary covariance matrix (symmetric positive definite)

Numerical Stability
-------------------
- Uses scipy.linalg.expm with RuntimeWarning suppression for ill-conditioned cases
- Falls back to stationary covariance when matrix exponential produces non-finite values
- Projects Q and P to positive definite via project_pd() to handle numerical errors
- Regularizes Lyapunov solution with eps*I for robustness

Implementation Notes
--------------------
- The Lyapunov identity Q = P0 - Phi P0 Phi^T requires F to be stable (Hurwitz)
  but doesn't enforce it explicitly; it replaces the Van Loan block-matrix
  extraction, which overflows at the large gaps in this cadence (module docstring)
- For dt extremely large relative to system time constants, the large-Δt shortcut
  provides accurate results without exponential of large negative numbers
- The stationary covariance solution assumes F is Hurwitz (all eigenvalues with
  negative real parts) - check stability separately if needed
- Q is projected to positive definite to account for numerical errors in expm

Dependencies
------------
- numpy for linear algebra operations
- scipy.linalg.expm for matrix exponential
- scipy.linalg.solve_continuous_lyapunov for Lyapunov equation
- .optimizer_utils.project_pd for positive definiteness projection

See Also
--------
- simulate() in mcarma.simulate uses these functions for exact simulation
- kalman_filter() in mcarma.filter uses these for prediction steps
- optimizer_utils.build_state_space constructs F, G, H for MCARMA models

Raises
------
- May raise np.linalg.LinAlgError if F has eigenvalues with non-negative real
  parts and the Lyapunov equation is solved (stationary_cov assumes stability)
- Scipy's expm may raise for highly ill-conditioned matrices, but warnings are
  suppressed and fallback behavior is triggered

Usage Examples (Internal)
-------------------------
Typical usage within MCARMA pipeline:

    # Build state-space from AR/MA parameters
    F, G, H, V = build_state_space(ar_factors, ma_factors, V, d, p, q)
    
    # Compute stationary initialization
    P0 = stationary_cov(F, G, V)
    x0 = np.random.multivariate_normal(np.zeros(state_dim), P0)
    
    # Propagate through irregular observations
    for k in range(1, n):
        dt = t_obs[k] - t_obs[k-1]
        Phi, Qd = transition_and_noise(F, G, V, dt)
        x = Phi @ x + np.random.multivariate_normal(np.zeros(state_dim), Qd)
        # ... observation step ...
"""

import warnings
import numpy as np
from scipy.linalg import expm, solve_continuous_lyapunov
from .optimizer_utils import project_pd


def transition_and_noise(F, G, V, dt,
                         lambda_max_real=None, P_stationary=None):
    """
    Compute exact discrete-time state transition Phi and integrated process
    noise covariance Q via the stationary Lyapunov identity
    Q = P0 - Phi P0 Phi^T (NOT the Van Loan block-matrix extraction, which
    overflows at large gaps -- see the module docstring and docs/METHODS.Rmd §3).

    Discretizes:
        dZ = F Z dt + G dB,   Cov(dB) = V dt

    Parameters
    ----------
    F : (n, n) drift matrix
    G : (n, d) input matrix
    V : (d, d) latent driving covariance
    dt : float, time increment
    lambda_max_real : float or None
        Pre-computed max real part of eigenvalues of F.
    P_stationary : (n, n) or None
        Pre-computed stationary covariance, used when dt is very large.

    Returns
    -------
    Phi : (n, n) state transition matrix
    Q   : (n, n) integrated process noise covariance (symmetric PD)
    """
    n = F.shape[0]

    if lambda_max_real is None:
        lambda_max_real = float(np.max(np.real(np.linalg.eigvals(F))))

    P0 = (P_stationary if P_stationary is not None
          else project_pd(stationary_cov(F, G, V)))

    # Large-dt shortcut: Phi -> 0, Q -> stationary covariance.
    eps = 1e-10
    if lambda_max_real < 0 and np.exp(lambda_max_real * dt) < eps:
        return np.zeros((n, n)), P0.copy()

    # Phi = exp(F*dt) computed directly. NB: do NOT extract Phi/Q from the Van
    # Loan augmented matrix [[-F, GVGT],[0, F^T]] — its top-left -F block is
    # *anti-stable* (eigenvalues -lambda_i, positive real part since F is Hurwitz),
    # so expm of it grows like e^{|Re lambda|*dt} and overflows for dt beyond a few
    # fast-mode time constants (e.g. ~1e39 at a 100-unit seasonal gap), corrupting
    # Q. The Lyapunov identity below uses only the bounded, stable Phi.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        Phi = expm(F * dt)

    # Overflow guard: non-finite Phi only when F is (near-)unstable, i.e. an
    # out-of-range parameter during optimisation; fall back to the sentinel.
    if not np.isfinite(Phi).all():
        return np.zeros((n, n)), P0.copy()

    # Exact integrated process-noise covariance for a stationary linear system:
    #     Qd(dt) = integral_0^dt e^{Fs} G V G^T e^{F^T s} ds = P0 - Phi P0 Phi^T.
    # This Lyapunov identity is numerically stable for all dt (it needs only the
    # bounded Phi and the stationary P0) and matches the defining integral; it
    # replaces the sign-flipped Van Loan extraction whose Qd was wrong even at
    # moderate dt and divergent at large dt.
    Q = P0 - Phi @ P0 @ Phi.T
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        Q = project_pd(0.5 * (Q + Q.T))
    return Phi, Q


def stationary_cov(F, G, V, eps=1e-12):
    """
    Compute the stationary state covariance P by solving the
    continuous Lyapunov equation:

        F P + P F^T + G V G^T = 0

    Parameters
    ----------
    F : (n, n) stable drift matrix (all eigenvalues negative real part)
    G : (n, d) input matrix
    V : (d, d) latent driving covariance
    eps : float, diagonal regularisation for numerical stability

    Returns
    -------
    P : (n, n) stationary covariance (symmetric positive definite)
    """
    Qc = G @ V @ G.T
    P  = solve_continuous_lyapunov(F, -Qc)
    P  = 0.5 * (P + P.T)
    P += eps * np.eye(F.shape[0])
    return P