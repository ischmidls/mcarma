"""
mcarma/smoother.py
==================
Rauch-Tung-Striebel (RTS) smoother for MCARMA state-space models.

See docs/METHODS.Rmd sec 4 (filter) - the smoother back-passes the same recursion.

Provides full-data smoothing (not just one-step-ahead prediction) for:
  - Smoothed residuals at observation times (model diagnostics)
  - Interpolation / forecasting at arbitrary evaluation times

The smoother runs a forward Kalman pass (identical in structure to
kalman.loglik) then a backward RTS pass, optionally on a merged grid
that includes user-supplied evaluation times.

Model Formulation
-----------------
Identical to kalman.py — see that module's docstring for the full
state-space equations. The smoother uses the same parameter convention:

    F, G, H, Sigma, C_list, R_list, mu

and the same precomputed-transition infrastructure (jax_transitions).

Smoother Equations (Backward Pass)
------------------------------------
After the forward Kalman filter produces filtered means x_{k|k} and
covariances P_{k|k}, the RTS backward recursion is, for k = n-2 ... 0:

    G_k   = P_{k|k} Φ_{k+1}ᵀ P_{k+1|k}⁻¹        [smoother gain]
    x_{k|n} = x_{k|k} + G_k (x_{k+1|n} - x_{k+1|k})
    P_{k|n} = P_{k|k} + G_k (P_{k+1|n} - P_{k+1|k}) G_kᵀ

P_{k+1|k}⁻¹ is computed via np.linalg.solve for numerical stability.
When Φ_{k+1} = I and Q_{k+1} = 0 (dt = 0), G_k = I exactly and the
recursion collapses to x_{k|n} = x_{k+1|n}, P_{k|n} = P_{k+1|n}.

Evaluation at Arbitrary Times
------------------------------
When t_eval is provided, evaluation times are merged with t_obs into a
single sorted grid t_all.  No observation update occurs at t_eval points
(obs_mask = False).  The forward/backward passes run on t_all, and
smoothed states are extracted at the eval-time indices.  This gives exact
Gaussian posterior means and variances at any requested time, including
forecasts outside [t_obs.min(), t_obs.max()].

Functions
---------
smooth(t_obs, y_obs, F, G, H, V, C_list, R_list, mu,
       t_eval=None, bands_eval=None, lambda_max_real=None)

    Run RTS smoother and return smoothed predictions.

    Parameters
    ----------
    t_obs       : (n,) observation times
    y_obs       : (n,) scalar observations (not pre-centred)
    F           : (dp, dp) drift matrix
    G           : (dp, d) input matrix
    H           : (d, dp) MA observation matrix
    V           : (d, d) latent driving covariance
    C_list      : list of n (1, d) band selector matrices
    R_list      : list of n (1, 1) observation noise matrices
    mu          : (d, 1) per-band mean vector
    t_eval      : (m,) optional evaluation times; may include times
                  outside [t_obs.min(), t_obs.max()] (forecasts).
                  If None, smoothed values are returned only at t_obs.
    bands_eval  : (m,) int array of band indices (0-based) for each
                  t_eval point.  Ignored when t_eval is None.
                  Defaults to all-zeros (band 0) if t_eval is given
                  but bands_eval is omitted.
    lambda_max_real : float or None
                  Pre-computed max real eigenvalue of F (passed to
                  transition_and_noise).  Computed internally if None.

    Returns
    -------
    result : dict with keys
        "t_obs"       : (n,) — same as input t_obs
        "yhat_smooth" : (n,) — smoothed latent mean  μ_k + C_eff_k x_{k|n}
        "std_state"   : (n,) — smoothed latent std from state uncertainty only
                                sqrt(C_eff_k P_{k|n} C_eff_k^T)
        "std_smooth"  : (n,) — smoothed observation std (state uncertainty + R)
                                sqrt(C_eff_k P_{k|n} C_eff_k^T + R_k)
        "residuals"   : (n,) — y_obs - yhat_smooth

        If t_eval is not None, also includes:
        "t_eval"      : (m,) — same as input t_eval
        "yhat_eval"   : (m,) — smoothed latent mean at eval times
        "std_eval"    : (m,) — smoothed latent std at eval times

    Returns None if the forward Kalman pass diverges (same semantics as
    loglik returning NaN).

Notes
-----
- Memory usage: O(n_total × state_dim²) for the stored P arrays.
  For dense t_eval grids (e.g. 10 000 points) with large state_dim
  this may be significant; consider chunking if needed.
- Symmetry of P is enforced at each step (same as kalman.py).
- When dt = 0 (simultaneous observations across bands), Φ = I, Q = 0
  and the smoother gain collapses to the identity — handled explicitly.
"""

import numpy as np
from .statespace import stationary_cov
from .optimizer_utils import project_pd
from .jax_transitions import precompute_transitions_jax


def smooth(t_obs, y_obs, F, G, H, V, C_list, R_list, mu,
           t_eval=None, bands_eval=None, lambda_max_real=None):
    """
    Rauch-Tung-Striebel smoother for MCARMA.  See module docstring for
    full parameter / return documentation.
    """
    n_obs     = len(t_obs)
    state_dim = F.shape[0]
    d         = H.shape[0]

    # ------------------------------------------------------------------
    # Stationary initialisation (same as kalman.loglik)
    # ------------------------------------------------------------------
    if lambda_max_real is None:
        lambda_max_real = float(np.max(np.real(np.linalg.eigvals(F))))
    P_stationary = project_pd(stationary_cov(F, G, V))

    # Note: the state prior has zero mean. The observation mean `mu` is
    # applied separately through the observation equation, so the smoothed
    # predictions are the posterior means conditional on the observed data.
    # This means the first smoothed point can differ from the unconditional
    # mean and may appear near zero if the initial observation deviates from
    # `mu` due to process or observation noise.
    
    # ------------------------------------------------------------------
    # Build merged time grid
    # ------------------------------------------------------------------
    # t_obs is assumed sorted (required by the Kalman filter).
    # We preserve ALL t_obs entries (including same-time multi-band repeats)
    # and insert any t_eval times that don't already appear in t_obs.
    if t_eval is not None:
        t_eval = np.asarray(t_eval, dtype=float)
        if bands_eval is None:
            bands_eval = np.zeros(len(t_eval), dtype=int)
        else:
            bands_eval = np.asarray(bands_eval, dtype=int)

        # Only insert eval times not already present in t_obs
        _eval_new = t_eval[~np.isin(t_eval, t_obs)]
        t_merged  = np.sort(np.concatenate([t_obs, _eval_new]))
    else:
        t_merged = t_obs.copy()

    n_total = len(t_merged)

    # ------------------------------------------------------------------
    # Map each merged-grid step to its observation index (or -1)
    # ------------------------------------------------------------------
    # t_merged is sorted and contains all t_obs values (with repeats).
    # Walk t_obs in order; for each time, claim the earliest unclaimed
    # position in t_merged with that value.
    obs_index = np.full(n_total, -1, dtype=int)   # -1 = no observation
    _next_slot = {}   # time -> next unclaimed position in t_merged
    for i, t in enumerate(t_merged):
        if t not in _next_slot:
            _next_slot[t] = i
    for orig_k, t in enumerate(t_obs):
        slot = _next_slot[t]
        obs_index[slot] = orig_k
        # Advance to the next occurrence of this time value in t_merged
        _next_slot[t] = slot + 1  # safe because t_merged has >= as many repeats as t_obs

    # ------------------------------------------------------------------
    # Precompute transitions on the merged grid
    # ------------------------------------------------------------------
    Phi_arr, Q_arr = precompute_transitions_jax(
        t_merged, F, G, V, lambda_max_real, P_stationary)

    # ------------------------------------------------------------------
    # Precompute per-observation effective observation vectors
    # ------------------------------------------------------------------
    C_eff_obs = np.array([C_list[k] @ H for k in range(n_obs)])[:, 0, :]  # (n_obs, state_dim)
    mu_obs    = np.array([(C_list[k] @ mu)[0, 0] for k in range(n_obs)])   # (n_obs,)
    R_obs     = np.array([R_list[k][0, 0] for k in range(n_obs)])           # (n_obs,)

    # ------------------------------------------------------------------
    # For eval times: build effective observation vectors (extraction only)
    # ------------------------------------------------------------------
    if t_eval is not None:
        eye_d = np.eye(d)
        C_eff_eval  = np.array([eye_d[b:b+1, :] @ H
                                 for b in bands_eval])[:, 0, :]   # (m, state_dim)
        mu_eval_arr = np.array([float(mu[b, 0]) for b in bands_eval])  # (m,)

        # For each t_eval, find its index in t_merged.
        # Use side='right'-1 so co-temporal obs-time entries pick the
        # last position (i.e. post-update state after all same-time obs).
        eval_merged_idx = np.searchsorted(t_merged, t_eval, side='right') - 1

    # ------------------------------------------------------------------
    # Storage for forward pass
    # ------------------------------------------------------------------
    x_pred = np.zeros((n_total, state_dim))
    P_pred = np.zeros((n_total, state_dim, state_dim))
    x_filt = np.zeros((n_total, state_dim))
    P_filt = np.zeros((n_total, state_dim, state_dim))

    # ------------------------------------------------------------------
    # FORWARD PASS  (Kalman filter — matches kalman.loglik scalar path)
    # ------------------------------------------------------------------
    x = np.zeros(state_dim)
    P = P_stationary.copy()
    eye = np.eye(state_dim)

    for i in range(n_total):
        # Prediction
        if i > 0:
            x = Phi_arr[i] @ x
            P = Phi_arr[i] @ P @ Phi_arr[i].T + Q_arr[i]
            P = 0.5 * (P + P.T)

        x_pred[i] = x
        P_pred[i] = P

        # Divergence check
        if not np.isfinite(P).all() or P.diagonal().min() < -1e-10:
            return None

        # Update (only at observation steps)
        k = obs_index[i]
        if k >= 0:
            c  = C_eff_obs[k]
            v  = y_obs[k] - mu_obs[k] - c @ x
            Pc = P @ c
            S  = float(c @ Pc) + R_obs[k]

            if not np.isfinite(S) or S <= 0.0:
                return None

            K   = Pc / S
            x   = x + K * v
            IKC = eye - np.outer(K, c)
            P   = IKC @ P @ IKC.T + R_obs[k] * np.outer(K, K)
            P   = 0.5 * (P + P.T)

            if not np.isfinite(P).all() or P.diagonal().min() < -1e-10:
                return None

        x_filt[i] = x
        P_filt[i] = P

    # ------------------------------------------------------------------
    # BACKWARD PASS  (RTS smoother)
    # ------------------------------------------------------------------
    x_smooth = x_filt.copy()
    P_smooth = P_filt.copy()

    for i in range(n_total - 2, -1, -1):
        Phi_next = Phi_arr[i + 1]   # transition from step i to i+1
        Q_next   = Q_arr[i + 1]

        # dt=0 case: Phi=I, Q=0 → smoother gain is I, states pass through
        dt_i = float(t_merged[i + 1] - t_merged[i])
        if dt_i == 0.0:
            x_smooth[i] = x_smooth[i + 1]
            P_smooth[i] = P_smooth[i + 1]
            continue

        # Smoother gain: G_i = P_filt[i] Phi_next^T  P_pred[i+1]^{-1}
        # Use solve instead of explicit inverse for stability
        # G_i^T = solve(P_pred[i+1], Phi_next @ P_filt[i]^T) = solve(P_pred[i+1], Phi_next @ P_filt[i])
        try:
            # solve: P_pred[i+1] X = Phi_next @ P_filt[i]
            # → X^T = G_i
            rhs = Phi_next @ P_filt[i]                         # (state_dim, state_dim)
            Gain = np.linalg.solve(P_pred[i + 1], rhs).T       # (state_dim, state_dim)
        except np.linalg.LinAlgError:
            # Singular P_pred — fall back to pseudo-inverse
            Gain = P_filt[i] @ Phi_next.T @ np.linalg.pinv(P_pred[i + 1])

        dx   = x_smooth[i + 1] - x_pred[i + 1]
        dP   = P_smooth[i + 1] - P_pred[i + 1]

        x_smooth[i] = x_filt[i] + Gain @ dx
        P_smooth[i] = P_filt[i] + Gain @ dP @ Gain.T
        P_smooth[i] = 0.5 * (P_smooth[i] + P_smooth[i].T)

    # ------------------------------------------------------------------
    # Extract smoothed values at observation times
    # ------------------------------------------------------------------
    # obs_index maps merged-grid position → original obs row index;
    # invert to get merged-grid position for each obs row.
    obs_merged_pos = np.zeros(n_obs, dtype=int)
    for i in range(n_total):
        k = obs_index[i]
        if k >= 0:
            obs_merged_pos[k] = i

    yhat_smooth = np.array([
        mu_obs[k] + C_eff_obs[k] @ x_smooth[obs_merged_pos[k]]
        for k in range(n_obs)
    ])
    # Smoothed observation variance = state uncertainty + observation noise
    # std_smooth includes both latent state uncertainty AND measurement noise R
    # State-only uncertainty (no observation noise)
    std_state = np.array([
        np.sqrt(max(C_eff_obs[k] @ P_smooth[obs_merged_pos[k]] @ C_eff_obs[k], 0.0))
        for k in range(n_obs)
    ])

    # Full observation uncertainty (state uncertainty + measurement noise)
    std_smooth = np.array([
        np.sqrt(max(C_eff_obs[k] @ P_smooth[obs_merged_pos[k]] @ C_eff_obs[k] + R_obs[k], 0.0))
        for k in range(n_obs)
    ])
    residuals = y_obs - yhat_smooth

    result = {
        "t_obs":        t_obs,
        "yhat_smooth":  yhat_smooth,
        "std_state":    std_state,
        "std_smooth":   std_smooth,
        "residuals":    residuals,
    }

    # ------------------------------------------------------------------
    # Extract smoothed values at evaluation times
    # (Note: eval times typically don't have observations, so R=0 for them)
    # ------------------------------------------------------------------
    if t_eval is not None:
        yhat_eval = np.array([
            mu_eval_arr[j] + C_eff_eval[j] @ x_smooth[eval_merged_idx[j]]
            for j in range(len(t_eval))
        ])
        # Eval times have no observation noise (R=0) unless they coincide with t_obs
        # For now, report only state uncertainty; R contribution is zero
        std_eval = np.array([
            np.sqrt(max(C_eff_eval[j] @ P_smooth[eval_merged_idx[j]] @ C_eff_eval[j], 0.0))
            for j in range(len(t_eval))
        ])
        result["t_eval"]    = t_eval
        result["yhat_eval"] = yhat_eval
        result["std_eval"]  = std_eval

    return result
