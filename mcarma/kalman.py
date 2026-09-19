"""
mcarma/kalman.py
================
Kalman filter for MCARMA state-space models with irregular observations.

See docs/METHODS.Rmd sec 4 (prediction-error log-likelihood).

Provides efficient log-likelihood computation via prediction error decomposition:
  1. loglik() - Main log-likelihood function with performance optimizations
  2. predict() - Standalone Kalman prediction step
  3. update() - Standalone Kalman update step (Joseph form)

The Kalman filter handles irregularly spaced observations through exact
discrete-time transitions (precomputed per unique time gap) and supports
scalar observations (single band at each time point) with band-specific
observation matrices.

Model Formulation
-----------------
Continuous-time state equation (zero-mean):
    dZ(t) = F Z(t) dt + G dB(t),   Cov(dB(t)) = Σ dt
    Z(t₀) ~ N(0, P∞)  [stationary distribution]

Discrete-time transitions between observation times:
    Z_k = Φ(Δt_k) Z_{k-1} + w_k,   w_k ~ N(0, Q_d(Δt_k))
    where Φ(Δt) = exp(F·Δt) and Q_d(Δt) = ∫₀^{Δt} Φ(s) G Σ Gᵀ Φ(s)ᵀ ds

Observation equation (scalar at each time):
    Y_k = C_k (μ + H Z_k) + ε_k,   ε_k ~ N(0, R_k)

Here C_k is a 1×d selector matrix (one row from identity), so only one
band is observed at time t_k. Multiple bands can be observed at the
same timestamp (dt=0), handled by sequential updates with zero time gap.

Kalman Filter Equations (Prediction Error Decomposition)
---------------------------------------------------------
Prediction:
    x_{k|k-1} = Φ_k x_{k-1|k-1}
    P_{k|k-1} = Φ_k P_{k-1|k-1} Φ_kᵀ + Q_k

Innovation:
    ν_k = Y_k - C_k μ - C_k H x_{k|k-1}
    S_k = C_k H P_{k|k-1} (C_k H)ᵀ + R_k  [scalar]

Update (Joseph form for numerical stability):
    K_k = P_{k|k-1} (C_k H)ᵀ / S_k
    x_{k|k} = x_{k|k-1} + K_k ν_k
    P_{k|k} = (I - K_k C_k H) P_{k|k-1} (I - K_k C_k H)ᵀ + K_k R_k K_kᵀ

Log-likelihood contribution at time k:
    ℓ_k = -½ [log(2π) + log(S_k) + ν_k² / S_k]

Total log-likelihood: ℓ = Σ_k ℓ_k

Performance Optimizations
-------------------------
1. Transition matrix caching:
   - Rounds dt to 4 decimal places for cache key
   - Reuses (Φ, Q) for identical time gaps
   - Handles dt=0 as special case (Φ=I, Q=0)

2. Precomputation before main loop:
   - C_eff = C_k H for all k (pre-multiplied observation matrices)
   - μ_k = C_k μ for all k (precomputed means)
   - R_k extracted to flat array

3. Scalar arithmetic:
   - S_k is always scalar (single band observed at each time)
   - Replaces matrix inversion with simple division
   - Uses float operations instead of matrix algebra

4. Symmetry enforcement:
   - P = 0.5*(P + Pᵀ) after each operation
   - Joseph form maintains symmetry numerically

5. Early termination:
   - Returns NaN for non-finite S, P, or ll
   - Rejects negative diagonal entries (numerical instability)
   - Avoids project_pd masking (indicates bad parameter region)

Mathematical Details
--------------------
Stationary Initialization:
    P∞ solves Lyapunov equation: F P∞ + P∞ Fᵀ + G Σ Gᵀ = 0
    x₀ = 0 (zero-mean process)

Transition Computation:
    Uses Van Loan method (via transition_and_noise from statespace.py)
    For large dt with exp(λ_max·dt) < eps: Φ ≈ 0, Q ≈ P∞

Joseph Form Update:
    Numerically stable for ill-conditioned problems
    Preserves positive semidefiniteness analytically
    Avoids subtraction of positive semidefinite matrices

Functions
---------
loglik(t_obs, y_obs, F, G, H, Sigma, C_list, R_list, mu, slopes=None,
       print_every=60, lambda_max_real=None, return_pred=False)
    Compute log-likelihood via Kalman filter with prediction error decomposition.

    Parameters:
        t_obs : ndarray (n,) - Observation times
        y_obs : ndarray (n,) - Scalar observations (not pre-centred)
        F : ndarray (dp, dp) - State transition matrix (continuous-time)
        G : ndarray (dp, d) - Noise input matrix
        H : ndarray (d, dp) - Observation matrix (MA part)
        Sigma : ndarray (d, d) - Innovation covariance
        C_list : list of n arrays (1, d) - Band selector matrices
        R_list : list of n arrays (1, 1) - Observation noise variances
        mu : ndarray (d, 1) - Per-band mean vector
        print_every : float - Seconds between progress prints (0 to disable)
        lambda_max_real : float or None - Precomputed max real eigenvalue of F
        return_pred : bool - If True, also return yhat (n,) one-step-ahead
                             predicted observations. Default False.

    Returns:
        ll : float - Log-likelihood value (NaN if numerically degenerate)
        yhat : ndarray (n,) - One-step-ahead predicted observations
               (only returned when return_pred=True)

predict(x, P, Phi, Q)
    Standalone Kalman prediction step.

    Parameters:
        x : ndarray (n,) or (n,1) - Current state mean
        P : ndarray (n,n) - Current state covariance
        Phi : ndarray (n,n) - Transition matrix
        Q : ndarray (n,n) - Process noise covariance

    Returns:
        x_pred : ndarray (n,) - Predicted state mean
        P_pred : ndarray (n,n) - Predicted state covariance

update(x, P, v, C_eff, R)
    Standalone Kalman update step (Joseph form).

    Parameters:
        x : ndarray (n,) - Predicted state mean
        P : ndarray (n,n) - Predicted state covariance
        v : scalar, (1,), or (1,1) - Innovation (observation - prediction)
        C_eff : ndarray (state_dim,) - Effective observation matrix (C_k @ H)
        R : scalar, (1,), or (1,1) - Observation noise variance

    Returns:
        x_upd : ndarray (n,) - Updated state mean
        P_upd : ndarray (n,n) - Updated state covariance
        S : ndarray (1,1) - Innovation covariance (for API compatibility)

Implementation Notes
--------------------
- Assumes single band observed at each time (C_k is 1×d selector)
- Supports multiple observations at same timestamp (dt=0 handled specially)
- State dimension n = d·p (lag-major ordering)
- Uses Joseph form for numerical stability in update step
- Does NOT use project_pd to mask bad parameters (returns NaN instead)
- Progress printing reports observation count and elapsed time

Numerical Considerations
------------------------
- Early rejection of non-finite values prevents optimization from exploring
  numerically degenerate regions
- Symmetry averaging (0.5*(P+P.T)) reduces floating-point asymmetry
- Diagonal negativity check (P.diagonal().min() < -1e-10) detects loss of
  positive semidefiniteness before numerical errors propagate
- Transition caching reduces expm calls by ~20x for data with repeated gaps
- dt rounding to 4 decimals balances cache efficiency with accuracy

Dependencies
------------
- numpy for array operations
- time for progress timing
- warnings for suppressing expected numerical warnings
- .statespace.transition_and_noise for discrete-time conversion
- .statespace.stationary_cov for initial covariance
- .optimizer_utils.project_pd for emergency projection (update only)

Raises
------
- Returns NaN (not exception) for numerical degeneracy to allow optimizer
  to backtrack from invalid parameter regions
- Assumes input validation occurs in calling functions (fit.py, validate.py)

Performance Notes
-----------------
For typical use cases (n=200 observations, state_dim=20-50):
    - Precomputation: O(n·state_dim²) for C_eff and μ arrays
    - Main loop: O(n·state_dim²) for matrix operations
    - Transition caching: Reduces expm calls from O(n) to O(unique gaps)
    - Scalar S eliminates matrix inversion overhead
    - Joseph form adds O(state_dim²) per update (vs O(state_dim³) for standard form)
"""

import logging
import warnings
import numpy as np
import time
from .statespace import transition_and_noise, stationary_cov
from .optimizer_utils import project_pd

_log = logging.getLogger(__name__)


def loglik(t_obs, y_obs, F, G, H, V, C_list, R_list, mu,
           slopes=None, print_every=60, lambda_max_real=None,
           return_pred=False, return_innov=False):
    """
    Log-likelihood for linear Gaussian MCARMA state-space model via
    Kalman filter prediction error decomposition.

    State equation (zero-mean):
        dZ = F Z dt + G dB,   Cov(dB) = V dt
        Z_0 ~ stationary distribution (mean zero, cov P_stationary)

    Observation equation (optional deterministic per-band linear trend):
        Y_k = C_k (mu + beta * t_k + H Z_k) + eps_k,   eps_k ~ N(0, R_k)

    where ``beta`` is the per-band slope vector supplied via ``slopes``.
    With ``slopes=None`` the trend term vanishes and the observation
    equation reduces to ``Y_k = C_k (mu + H Z_k) + eps_k``.

    Innovation:
        v_k = Y_k - C_k mu - C_eff_k Z_{k|k-1}
        S_k = C_eff_k P_{k|k-1} C_eff_k^T + R_k   (always scalar)

    Performance
    -----------
    - All (Phi_k, Q_k) pairs and observation arrays are precomputed
      before the sequential loop, keeping the hot path free of scipy
      calls and Python list indexing overhead.
    - S is always scalar so matrix inversion is replaced by 1/S.
    - P symmetry is enforced by cheap averaging; project_pd is called
      only when a diagonal entry actually goes negative.

    Parameters
    ----------
    t_obs  : (n,) observation times
    y_obs  : (n,) scalar observations (NOT pre-centred)
    F      : (dp, dp) drift matrix
    G      : (dp, d) input matrix
    H      : (d, dp) MA observation matrix
    V      : (d, d) latent driving covariance
    C_list : list of n (1, d) band selector matrices
    R_list : list of n (1, 1) observation noise matrices
    mu     : (d, 1) per-band mean vector
    slopes : (d,) array or None, default None
        Per-band slopes of a known/fixed deterministic linear trend in
        observation time. When given, ``slopes[band_k] * t_obs[k]`` is added
        to the predicted observation mean (folded into the per-observation
        mean offset). When None, no trend is applied and behaviour is
        identical to the original function.
    print_every : seconds between progress prints (0 to disable)
    return_pred : bool, default False
        If True, also return yhat (n,) — the Kalman one-step-ahead
        predicted observations  E[Y_k | Y_1, ..., Y_{k-1}]  for every k.
        When False (default) behaviour is identical to the original function.

    Returns
    -------
    ll : float
        Log-likelihood (NaN if numerically degenerate).
    yhat : ndarray (n,), only when return_pred=True
        One-step-ahead predicted observations at each time step.
    """
    n_obs     = len(t_obs)
    state_dim = F.shape[0]

    # ------------------------------------------------------------------
    # Precompute once per loglik call
    # ------------------------------------------------------------------
    if lambda_max_real is None:
        # may not need this since neg_loglik() does an early check on F's eigenvalues,
        # but precompute here for safety and to avoid redundant computations in the Kalman loop.
        lambda_max_real = float(np.max(np.real(np.linalg.eigvals(F))))
    P_stationary    = project_pd(stationary_cov(F, G, V))

    # # Batch all transition matrices before the Kalman loop so expm is
    # # not called inside the sequential hot path.
    # #
    # # Cache by rounded dt: the quasar data has only ~35 unique gaps out
    # # of 686, so this reduces expm calls by ~20x. dt=0 (same-timestamp
    # # observations across bands) is handled as a special case: Phi=I, Q=0.
    # _transition_cache = {}
    # Phi_arr = [None] * n_obs
    # Q_arr   = [None] * n_obs
    # eye_state = np.eye(state_dim)
    # zero_Q    = np.zeros((state_dim, state_dim))

    # for k in range(1, n_obs):
    #     dt = float(t_obs[k] - t_obs[k - 1])
    #     if dt == 0.0:
    #         Phi_arr[k] = eye_state
    #         Q_arr[k]   = zero_Q
    #     else:
    #         # Round to 4 decimal places for cache key (sub-0.1 day precision)
    #         dt_key = round(dt, 4)
    #         if dt_key not in _transition_cache:
    #             _transition_cache[dt_key] = transition_and_noise(
    #                 F, G, Sigma, dt,
    #                 lambda_max_real=lambda_max_real,
    #                 P_stationary=P_stationary)
    #         Phi_arr[k], Q_arr[k] = _transition_cache[dt_key]

    # replace from  _transition_cache = {}  down to  Phi_arr[k], Q_arr[k] = ...
    from .jax_transitions import precompute_transitions_jax

    Phi_arr, Q_arr = precompute_transitions_jax(
        t_obs, F, G, V, lambda_max_real, P_stationary)

    # Precompute per-observation scalars/vectors
    C_eff_arr = np.array([C_list[k] @ H for k in range(n_obs)])[:, 0, :]  # (n, state_dim)
    mu_arr    = np.array([(C_list[k] @ mu)[0, 0] for k in range(n_obs)])  # (n,)
    R_arr     = np.array([R_list[k][0, 0] for k in range(n_obs)])          # (n,)

    # Optional fixed deterministic per-band linear trend, folded into the
    # per-observation mean offset so the sequential hot path stays unchanged.
    # band_k = argmax of the (1, d) selector C_list[k]; trend_k = slopes[band_k] * t_k.
    if slopes is not None:
        slopes   = np.asarray(slopes, dtype=float).ravel()
        band_idx = np.argmax(np.asarray(C_list)[:, 0, :], axis=1)          # (n,)
        mu_arr   = mu_arr + slopes[band_idx] * np.asarray(t_obs, dtype=float)

    # ------------------------------------------------------------------
    # Allocate predicted-observation array (always, cheaply)
    # ------------------------------------------------------------------
    yhat = np.zeros(n_obs)
    Sarr = np.full(n_obs, np.nan)   # one-step-ahead innovation variances (GOF)

    # ------------------------------------------------------------------
    # Sequential Kalman filter — scalar hot path
    # ------------------------------------------------------------------
    x   = np.zeros(state_dim)
    P   = P_stationary.copy()
    eye = np.eye(state_dim)

    ll   = 0.0
    t0   = time.time()
    last = t0

    # Suppress numpy overflow/invalid warnings inside the loop — these
    # occur when the optimizer explores degenerate parameters and are
    # handled explicitly by the isfinite checks below.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for k in range(n_obs):

            # Prediction
            if k > 0:
                x = Phi_arr[k] @ x
                P = Phi_arr[k] @ P @ Phi_arr[k].T + Q_arr[k]
                P = 0.5 * (P + P.T)

            # Innovation (scalar arithmetic)
            c      = C_eff_arr[k]               # (state_dim,)
            y_pred = mu_arr[k] + c @ x          # scalar
            yhat[k] = y_pred                    # store one-step-ahead prediction
            v      = y_obs[k] - y_pred          # scalar
            Pc     = P @ c                      # (state_dim,)
            S      = float(c @ Pc) + R_arr[k]  # scalar
            Sarr[k] = S

            if not np.isfinite(S) or S <= 0.0:
                if return_innov:
                    return float('nan'), yhat, Sarr
                return (float('nan'), yhat) if return_pred else float('nan')

            ll += -0.5 * (np.log(S) + v * v / S + np.log(2.0 * np.pi))

            if not np.isfinite(ll):
                if return_innov:
                    return float('nan'), yhat, Sarr
                return (float('nan'), yhat) if return_pred else float('nan')

            # Update — Joseph form with scalar K
            K   = Pc / S                                    # (state_dim,)
            x   = x + K * v
            IKC = eye - np.outer(K, c)
            P   = IKC @ P @ IKC.T + R_arr[k] * np.outer(K, K)

            # If P overflowed or went non-finite, this parameter set is
            # numerically degenerate — bail out immediately.
            if not np.isfinite(P).all():
                if return_innov:
                    return float('nan'), yhat, Sarr
                return (float('nan'), yhat) if return_pred else float('nan')

            P   = 0.5 * (P + P.T)

            # Initial condition  P stationary is PSD by construction.
            # Then prediction  P to Phi P Phi^T + Q$ (preserves PSD analytically)
            # update (Joseph form): also PSD analytically. So if P becomes indefinite,
            # it's due to numerical instability bad parameters (e.g. near-unstable
            # F, huge Q etc.) That means repairing with project_pd is masking a
            # bad region of parameter space rejecting is mathematically consistent with MLE
            if P.diagonal().min() < -1e-10:
                if return_innov:
                    return float('nan'), yhat, Sarr
                return (float('nan'), yhat) if return_pred else float('nan')

            if print_every > 0 and time.time() - last >= print_every:
                _log.info(f"[loglik] {k+1}/{n_obs} obs, "
                      f"{(time.time()-t0)/60:.1f} min elapsed")
                last = time.time()

    if return_innov:
        return float(ll), yhat, Sarr
    if return_pred:
        return float(ll), yhat
    return float(ll)


# ---------------------------------------------------------------------------
# Standalone predict / update — for external use (e.g. run_kalman in fit
# scripts). loglik() uses an inlined scalar version for performance.
# ---------------------------------------------------------------------------

def predict(x, P, Phi, Q):
    """Kalman prediction step. x: (n,) or (n,1), P: (n,n)."""
    x_pred = Phi @ x
    P_pred = Phi @ P @ Phi.T + Q
    P_pred = 0.5 * (P_pred + P_pred.T)
    return x_pred, P_pred


def update(x, P, v, C_eff, R):
    """
    Kalman update step (Joseph form).

    v, R may be scalar, (1,), or (1,1). Returns S as (1,1) for
    API compatibility with older fit scripts.
    """
    c  = np.asarray(C_eff).ravel()
    r  = float(np.asarray(R).ravel()[0])
    vi = float(np.asarray(v).ravel()[0])

    Pc  = P @ c
    S   = float(c @ Pc) + r
    K   = Pc / S
    x_upd = x + K * vi

    IKC   = np.eye(P.shape[0]) - np.outer(K, c)
    P_upd = IKC @ P @ IKC.T + r * np.outer(K, K)
    P_upd = 0.5 * (P_upd + P_upd.T)

    if P_upd.diagonal().min() < 0.0:
        P_upd = project_pd(P_upd)

    return x_upd, P_upd, np.array([[S]])