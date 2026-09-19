"""
Maximum-likelihood fitting for multiband continuous-time ARMA models.

See docs/METHODS.Rmd sec 6-7 for the penalty terms and the penalized-Hessian
standard errors. Those terms are scientifically motivated regularizers on a
penalized maximum-likelihood objective. They are not a Bayesian prior and
nothing here reports a posterior.

This module provides optimisation utilities for MCARMA(p, q) models under
the continuous-time state-space parameterization used throughout the package.

Parameterization
----------------
AR and MA polynomial factors are represented in unconstrained log-space using
the Jones-style transform implemented in `optimizer_utils.py`. Positive
coefficients are exponentiated during unpacking, allowing optimisation over
R^n while preserving admissible polynomial structure.

For each band:

- linear factors are parameterized by log(r),
- quadratic factors are parameterized by
      log(2 r), log(r^2 + omega^2),

where r > 0 is the root real part and omega is the oscillation frequency.

Process covariance matrices are parameterized through the Cholesky factor
Sigma = L L^T with log-variance contributions on the diagonal entries.

The resulting parameter vector is

    theta = [theta_AR,
             theta_MA,
             theta_cholesky,
             mu]

where mu contains per-band mean levels.

Cholesky parameterization strategy
-----------------------------------
Following R's cts package and YUIMA, the diagonal of the Cholesky factor L
is parameterized in log-variance space (log(L_ii^2)), while off-diagonal
entries remain unconstrained. This separation provides:

1. Identifiability for the innovation scale: the likelihood is invariant
   under L_ii -> lambda L_ii when sigma_jl scales by lambda^2 (scale-free
   for MA/AR).

2. The log-Cholesky parameterization guarantees positive-definite innovation
   covariance matrices while allowing unconstrained optimization in R^n, so
   positivity needs no hard bound. It does not stop log Sigma_bb from running
   to -inf near a likelihood degeneracy (pole-zero near-cancellation), and
   production fits therefore also carry a SOFT box on the log-variance
   diagonal: `chol_ridge_lambda` / `chol_ridge_center` /
   `chol_ridge_center_hi` in `neg_loglik`, constructed by
   `priors.sigma_softbox`. The box is a penalty rather than a constraint and
   is flat between its floor and its ceiling, so variances the data pin down
   are left unbiased and only unidentified directions are lifted.

3. Boundary compliance: log(L_ii^2) in R and exp(0.5 * .) -> L_ii > 0 without
   ad-hoc constraints or barriers.

Reference implementations:
  - R cts::mle() uses `log_sigma2` for the variance of the driving noise.
  - YUIMA::setYuima() constrains only the diagonal of the diffusion matrix.

Identifiability issues
----------------------
Degeneracy in the fitted covariance (e.g. near-zero innovation variance) is a
likelihood-geometry issue, not a parameterization issue. It arises from
instability, NaN Kalman likelihoods, or convergence into a covariance-degenerate
basin (typically driven by pole-zero near-cancellation). Positivity is
guaranteed by the log-Cholesky parameterization, so no hard variance floor is
imposed; the soft box described above is what keeps log Sigma_bb finite, and
degeneracy is diagnosed by likelihood evaluation and profile scans rather than
by clamping. A fit whose log-variance settles on the floor is at a constrained
optimum, which matters when reading its curvature: see
`inference.polished_cov`.

Optimisation
------------
Likelihood maximization is performed with repeated local BFGS restarts, with
an optional dual-annealing global search (`use_dual_annealing`) and an optional
bounded particle-swarm polish seeded at the best local optimum (`pso_polish`).
Invalid state-space realizations or unstable drift matrices are penalized
during optimisation.

Gradients are finite-difference by default. With `use_jax_grad=True` the
objective and its gradient come from `jax_loglik.py`, which is the path
production fits use; the drivers record which one actually ran in a
`grad_backend` field, and that field, not this docstring, is the record.
"""

import logging

import numpy as np
from scipy.optimize import minimize, dual_annealing
from .optimizer_utils import (pack_params_jones, build_state_space, unpack_params_jones,
                              _ar_roots_from_factor, _ma_roots_from_factor)
from .kalman import loglik
from .observation import ObservationData

_log = logging.getLogger(__name__)


def prepare_initial_params(data, p, q, random_state=None,
                            zeta_range=(0.05, 0.85), rho_min=0.5,
                            c_cadence=0.7, c_span=0.2,
                            ma_above_ar=False,
                            zeta_mode="under", zeta_over_range=(2.0, 6.0)):
    """
    Generate randomized initial parameters for MCARMA(p, q) optimisation.

    Parameters are produced directly in the unconstrained log-space required
    by the likelihood optimiser, so they can be passed directly to the
    solver without further transformation.

    AR initialisation
    -----------------
    The AR polynomial of each band is parameterised as a product of stable
    quadratic and/or linear factors (Section 2.1).  Quadratic factors are
    constructed in the frequency-damping pair ``(omega_n, zeta)``:

        root = -omega_n * (zeta +/- i * sqrt(1 - zeta^2))

    * ``omega_n`` is sampled log-uniformly from the **observable band**
      ``[omega_n_min, omega_n_max]``, where

        omega_n_min = 1 / (c_span * T_span)
        omega_n_max = 1 / (c_cadence * median(dt))

      with ``T_span = max(t_obs)-min(t_obs)`` and ``dt = t_i - t_{i-1}``.
    * ``zeta`` is drawn uniformly from ``zeta_range`` (default [0.05, 0.85])
      subject to the **same damping floor the penalty enforces**: the pole
      decay rate ``|Re lambda| = zeta * omega_n`` must be at least
      ``delta_min = 1/(c_span * T_span)`` (the 2c prior). This is imposed by
      clamping the lower zeta to ``max(zeta_range[0], delta_min/omega_n)``.
      Initialization and regularization therefore use ONE constraint set
      (Zhirui, 6/26): the band prior 2b on ``omega_n`` and the damping floor 2c
      on ``|Re lambda|``. (The previous init used different rules -- a 3-period
      floor ``|Re lambda| >= 3/T`` and a persistence ceiling
      ``|Re lambda| <= -log(rho_min)/dt`` -- which matched neither the prior's
      floor nor had any prior counterpart; ``rho_min`` is now unused.)
    * Linear AR factors (when `p` is odd) are treated as a real pole
      at ``-omega_n`` with ``omega_n`` sampled log-uniformly from the
      observable band.

    The resulting parameters are stored as logarithms:
      - for each quadratic factor: ``log(2 * zeta * omega_n)``  (log a1),
        ``log(omega_n^2)``                                     (log a2);
      - for a linear factor: ``log(omega_n)``                   (log of the
        decay rate).

    MA initialisation
    -----------------
    MA roots are placed near the AR roots to avoid extreme pole-zero
    separations while staying in the minimum-phase region.  This is done
    through a **relative frequency** parametrisation ``eta``:

        eta = (1 / |MA zero|) / omega_n_ref

    where ``omega_n_ref`` is the mean of the ``omega_n`` values used for
    the quadratic AR factors of the same band.

    * ``eta`` is sampled log-uniformly in
      ``[omega_n_min / omega_n_ref, omega_n_max / omega_n_ref]``,
      guaranteeing that the MA zero frequency stays inside the observable
      band.
    * For a linear MA factor: ``b_ma = 1 / (eta * omega_n_ref)``,
      stored as ``log(b_ma)``.
    * For quadratic MA factors: the real part ``r_ma`` is set via
      ``r_ma = eta * omega_n_ref`` and the imaginary part is chosen
      independently as ``omega_ma = r_ma * exp(Uniform(log 0.5, log 3.0))``.
      The stored parameters are ``log(2 * r_ma)`` and
      ``log(r_ma^2 + omega_ma^2)``.

    Covariance initialisation
    -------------------------
    The latent driving covariance matrix is represented by the elements of
    its Cholesky factor.  Diagonal entries are initialised around
    ``0.1 * empirical_std(y_obs)`` with a small random perturbation,
    transformed to log-space as ``2 * log(value)`` to ensure positivity.
    Off-diagonal entries are set to small random normal values.

    Mean initialisation
    -------------------
    Per-band means are initialised as the empirical average of the
    observations in each band.

    Parameters
    ----------
    data : ObservationData
        Observed multi-band time series with attributes ``t_obs`` (array),
        ``y_obs`` (array), ``band`` (array of band indices), and ``d`` (int).
    p, q : int
        AR and MA orders.
    random_state : int, Generator, or None
        Random seed or NumPy random generator.
    zeta_range : tuple (float, float), optional
        Range for the damping ratio ``zeta``.  Only affects quadratic AR
        factors (default (0.05, 0.85)).
    rho_min : float, optional
        Minimum discrete-time persistence
        ``exp(-zeta * omega_n * dt_med)`` (default 0.5).
    c_cadence : float, optional
        Fraction of the median cadence that defines the fastest resolvable
        frequency (default 0.7; fit() uses 0.7 for both this init draw and the
        matching band prior _band_prior_neglogp so the launch region and the
        penalty edges agree).
    c_span : float, optional
        Fraction of the total time span that defines the slowest resolvable
        frequency (default 0.2).

    Returns
    -------
    ndarray
        Unconstrained parameter vector suitable for direct optimisation.
        The ordering is: ``[ar_params, ma_params, chol_params, mu_params]``.
    """
    rng = (random_state if isinstance(random_state, np.random.Generator)
           else np.random.default_rng(random_state))

    d      = data.d
    # Observable band MUST match the band prior (_band_prior_neglogp) and the
    # truth design (run_task dt_perband), or the warm start places AR poles / MA
    # zeros outside the band the prior then penalizes. Two prior bugs lived here:
    #   * T was hardcoded to 200 (ignoring the real span), and
    #   * dt_med used the GLOBAL interleaved diff, which is ~d x finer than what
    #     any single band sees -> omega_n_max ceiling ~d x too high.
    # Both inflate the ceiling, so the ma_above_ar MA zero initialised at |root| ~
    # eta*omega_n_ref landed FAR above Nyquist (|root|~2 vs true ceiling ~0.34),
    # at a likelihood below the (2,0) nesting floor -> the (2,1) collapse. Use the
    # real span and the COARSEST per-band median cadence (resolvability is a
    # per-band property).
    t_obs  = np.asarray(data.t_obs, float)
    band   = np.asarray(data.band)
    T_span = float(t_obs.max() - t_obs.min()) if t_obs.size > 1 else 1.0
    perband_dt = [np.median(np.diff(np.sort(t_obs[band == b])))
                  for b in np.unique(band) if int(np.sum(band == b)) > 1]
    if perband_dt:
        dt_med = float(max(perband_dt))
    else:
        dd = np.diff(np.sort(t_obs))
        dt_med = float(np.median(dd)) if dd.size else 0.0
    n_chol = d * (d + 1) // 2

    # ------------------------------------------------------------------
    # Observable-band endpoints — same derivation as find_identifiable_*
    # ------------------------------------------------------------------
    # Slowest resolvable: one full period fits in c_span * T_span
    # Fastest resolvable: Nyquist-like cap from cadence
    _safe_T    = T_span  if T_span  > 0.0 else 1.0
    _safe_dt   = dt_med  if dt_med  > 0.0 else _safe_T / max(data.n - 1, 1)
    omega_n_min = 1.0 / (c_span   * _safe_T)
    omega_n_max = 1.0 / (c_cadence * _safe_dt)
    # Guard: if band is degenerate fall back to old T-relative range
    if omega_n_min >= omega_n_max:
        omega_n_min = 0.5  / _safe_T
        omega_n_max = 20.0 / _safe_T

    has_linear_ar = p % 2
    num_quads_ar  = p // 2
    has_linear_ma = q % 2
    num_quads_ma  = q // 2

    # ------------------------------------------------------------------
    # AR initialisation
    # ar_omega_n_per_band[b] = list of omega_n drawn for band b's quad factors
    # (used below when initialising MA via the eta parameterisation)
    # ------------------------------------------------------------------
    ar_init              = []
    ar_omega_n_per_band  = []   # list[list[float]]

    for b in range(d):
        band_log_params = []
        band_omega_n    = []   # omega_n values for MA eta bounds

        # Linear AR factor: root at -r; sample r in the observable band
        if has_linear_ar:
            r = float(np.exp(
                rng.uniform(np.log(omega_n_min), np.log(omega_n_max))
            ))
            band_log_params.append(np.log(r))
            band_omega_n.append(r)   # treat as omega_n for MA purposes

        # Quadratic AR factors: (omega_n, zeta) drawn under the SAME constraint
        # set the penalty regularizes (Zhirui 6/26: one constraint set for
        # regularization AND initialization). Both constraints are shared with
        # neg_loglik's priors, not re-invented here:
        #   (2b) band prior  -> pole modulus omega_n in [omega_n_min, omega_n_max]
        #   (2c) damping floor-> |Re lambda| = zeta*omega_n >= delta_min,
        #                        delta_min = 1/(c_span*T_span) (== omega_n_min).
        # The old init used DIFFERENT rules (a 3-period floor |Re lambda| >= 3/T
        # and a persistence ceiling |Re lambda| <= -log(rho_min)/dt): the floor
        # constant disagreed with the prior's 5/T and the ceiling had no prior
        # counterpart. Unifying makes the warm start live in exactly the region
        # the penalized objective does not push against.
        delta_min = omega_n_min                       # = 1/(c_span * T_span)
        for _ in range(num_quads_ar):
            # meeting 7/23 #2: the random start can seed the OVERDAMPED basin
            # too, not only the underdamped one, so order selection is not
            # biased by an underdamped-only init. zeta_mode "under" (default)
            # reproduces the historical draw; "over" always draws zeta>1 with
            # omega_n biased to the slow end (real overdamped poles are slow,
            # which keeps the fast pole omega_n*(zeta+sqrt(zeta^2-1)) inside the
            # band); "both" flips a coin per quadratic factor. Mirrors the
            # _overdamped_ar_reset seed used on the warm-start path.
            draw_over = (zeta_mode == "over"
                         or (zeta_mode == "both" and rng.random() < 0.5))
            if draw_over:
                omega_n = float(np.exp(rng.uniform(
                    np.log(omega_n_min),
                    np.log(min(5.0 * omega_n_min, omega_n_max)))))
                zeta = float(rng.uniform(*zeta_over_range))
            else:
                omega_n = float(np.exp(
                    rng.uniform(np.log(omega_n_min), np.log(omega_n_max))
                ))
                # Damping floor 2c: require zeta*omega_n >= delta_min, i.e.
                # zeta >= delta_min/omega_n, clamped into the fit-reachable band.
                zeta_lo = min(max(zeta_range[0], delta_min / omega_n), zeta_range[1])
                zeta = float(rng.uniform(zeta_lo, zeta_range[1]))

            r = zeta * omega_n                    # real part of poles = a1/2
            band_log_params.append(np.log(2.0 * r))     # log(a1)
            band_log_params.append(np.log(omega_n ** 2)) # log(a2)
            band_omega_n.append(omega_n)

        ar_init.extend(band_log_params)
        ar_omega_n_per_band.append(band_omega_n)

    ar_init = np.array(ar_init)

    # ------------------------------------------------------------------
    # MA initialisation via eta parameterisation
    # eta = (1/b) / omega_n_ref  so that the MA zero stays in the
    # observable band; sample eta log-uniformly in [omega_n_min/omega_n_ref,
    # omega_n_max/omega_n_ref].
    # For quadratic MA factors we reuse the same idea on the real part.
    # ------------------------------------------------------------------
    ma_init = []
    if q > 0:
        for b in range(d):
            # Reference omega_n: use the mean of the quad AR values for this band
            band_omega_n = ar_omega_n_per_band[b]
            omega_n_ref  = (float(np.mean(band_omega_n))
                            if band_omega_n else 1.0 / _safe_T)

            eta_lo = omega_n_min / omega_n_ref
            eta_hi = omega_n_max / omega_n_ref
            # Ensure a non-degenerate log interval
            if eta_lo <= 0.0 or eta_lo >= eta_hi:
                eta_lo = 1e-3
                eta_hi = max(eta_hi, eta_lo * 10.0)
            # MA-above-AR option: the MA zero modulus is eta * omega_n_ref, so
            # eta >= 1 starts the MA feature at or above the AR natural
            # frequency (PSD argument: the MA boost acts above the pole). This
            # also keeps the warm start out of the eta~1 pole-zero cancellation
            # basin where (2,1) collapses to (2,0).
            #
            # CRUCIAL: the MA zero must still stay INSIDE the observable band
            # (|zero| <= omega_n_max). When omega_n_ref is already near the band
            # ceiling, "above the AR pole" and "in band" conflict; the old code
            # expanded eta_hi = eta_lo*10, placing the zero at |root| ~ 10*omega_n_ref
            # FAR above Nyquist. There the warm-start ll sits below the (2,0)
            # nesting floor and L-BFGS stalls -> the (2,1) fit converges BELOW
            # (2,0) (the observed collapse: ll=-283 < -74, b_hat ~ 1 vs true ~9).
            # Prefer in-band: clamp eta DOWN into the top of the band rather than
            # pushing the zero out of it.
            if ma_above_ar:
                eta_lo = max(eta_lo, 1.0)
                if eta_lo >= eta_hi:
                    # cannot be both above the AR pole AND in band -> keep in band,
                    # placing the zero high in the observable window [0.5,1]*eta_hi
                    eta_lo = max(eta_hi * 0.5, omega_n_min / omega_n_ref)

            # Linear MA factor: 1 + b_ma*z,  zero at -1/b_ma
            if has_linear_ma:
                eta   = float(np.exp(
                    rng.uniform(np.log(eta_lo), np.log(eta_hi))
                ))
                b_ma  = 1.0 / (eta * omega_n_ref)
                ma_init.append(np.log(b_ma))

            # Quadratic MA factors: same eta idea on the real part r_ma,
            # then choose imaginary part independently within the band
            for _ in range(num_quads_ma):
                eta_r  = float(np.exp(
                    rng.uniform(np.log(eta_lo), np.log(eta_hi))
                ))
                r_ma   = eta_r * omega_n_ref
                # imaginary part: sample ratio omega_ma/r_ma in [0.5, 3]
                omega_ma = r_ma * float(np.exp(
                    rng.uniform(np.log(0.5), np.log(3.0))
                ))
                ma_init.append(np.log(2.0 * r_ma))
                ma_init.append(np.log(r_ma ** 2 + omega_ma ** 2))

    ma_init = np.array(ma_init) if ma_init else np.array([])

    # ------------------------------------------------------------------
    # Cholesky and mu — unchanged
    # ------------------------------------------------------------------
    emp_std = np.std(data.y_obs)
    if emp_std < 1e-8:
        emp_std = 1.0
    rows, cols = np.tril_indices(d)
    chol_init  = np.zeros(n_chol)
    for k in range(n_chol):
        i, j = int(rows[k]), int(cols[k])
        if i == j:
            raw_diag     = emp_std * 0.1 * (0.5 + 0.2 * abs(rng.standard_normal()))
            chol_init[k] = 2.0 * np.log(max(raw_diag, 1e-10))
        else:
            chol_init[k] = 0.01 * rng.standard_normal()

    mu_init = np.array([
        np.mean(data.y_obs[data.band == b]) if np.any(data.band == b) else 0.0
        for b in range(d)
    ])

    return np.concatenate([ar_init, ma_init, chol_init, mu_init])

def prepare_initial_params_safe(data, p, q, rng, max_tries=50,
                                ma_above_ar=False, zeta_mode="under"):
    """
    Generate a numerically valid initial parameter vector.

    Random initializations are repeatedly sampled until the resulting
    parameterization successfully produces a valid continuous-time
    state-space realization.

    A proposal is rejected if:

    - polynomial unpacking fails,
    - the state-space system cannot be constructed,
    - covariance parameterization becomes numerically invalid.

    Parameters
    ----------
    max_tries : int
        Maximum number of initialization attempts before failure.

    Returns
    -------
    ndarray
        Valid unconstrained optimization vector.

    Raises
    ------
    RuntimeError
        If no valid initialization is found.
    """
    for attempt in range(max_tries):
        theta = prepare_initial_params(data, p, q, random_state=rng,
                                        ma_above_ar=ma_above_ar,
                                        zeta_mode=zeta_mode)
        try:
            ar_f, ma_f, Sigma = unpack_params_jones(theta[:-data.d], data.d, p, q)
            build_state_space(ar_f, ma_f, Sigma, data.d, p, q)
            return theta
        except Exception as e:
            if attempt == 0:
                _log.warning(f"[init] First attempt failed: {e}")
            last_exc = e
    raise RuntimeError(
        f"Failed to generate valid initial theta after {max_tries} retries. "
        f"Last error: {last_exc}"
    )


def _underdamped_ar_reset(warm_theta, data, p, q, rng,
                          c_cadence=0.7, c_span=0.2,
                          zeta_range=(0.10, 0.40)):
    """Return a copy of ``warm_theta`` with the AR quadratic factors reset to an
    underdamped, low-frequency draw (keeping the MA / Cholesky / mu blocks).

    Motivation (the (2,1) collapse, diagnosed June 2026)
    ----------------------------------------------------
    Warm-starting (2,1) from a converged (2,0) fit inherits the (2,0) AR, which
    for (2,1)-generated data sits near the critical-damping fold (zeta ~ 1) with
    the pole at the observable ceiling -- a strong local optimum of the (2,0)
    surface. From there L-BFGS cannot reach the true UNDERDAMPED low-frequency
    pole (zeta ~ 0.15), so the (2,1) fit converges BELOW (2,0) with the MA
    collapsed. The truth basin is stable once entered (fit-from-truth is a clean
    optimum), so it suffices to ALSO probe it: reset the AR to a low-omega,
    low-zeta draw and let the optimizer descend. Only quadratic AR factors are
    reset (the oscillatory poles); any linear factor is left untouched.
    """
    theta = np.asarray(warm_theta, float).copy()
    d = data.d
    t_obs = np.asarray(data.t_obs, float)
    band = np.asarray(data.band)
    T_span = float(t_obs.max() - t_obs.min()) if t_obs.size > 1 else 1.0
    perband_dt = [np.median(np.diff(np.sort(t_obs[band == b])))
                  for b in np.unique(band) if int(np.sum(band == b)) > 1]
    dt_med = (float(max(perband_dt)) if perband_dt
              else (float(np.median(np.diff(np.sort(t_obs)))) if t_obs.size > 1 else 1.0))
    omega_n_min = 1.0 / (c_span * T_span) if T_span > 0 else 1e-3
    omega_n_max = 1.0 / (c_cadence * dt_med) if dt_med > 0 else 1.0
    if not (0.0 < omega_n_min < omega_n_max):
        omega_n_min, omega_n_max = 0.5 / max(T_span, 1.0), 20.0 / max(T_span, 1.0)
    # bias to the LOWER half of the observable band (truth poles are slow)
    log_lo, log_mid = np.log(omega_n_min), np.log(np.sqrt(omega_n_min * omega_n_max))

    num_quads_ar = p // 2
    has_linear_ar = p % 2
    for b in range(d):
        base = b * p
        for k in range(num_quads_ar):
            omega_n = float(np.exp(rng.uniform(log_lo, log_mid)))
            zeta = float(rng.uniform(*zeta_range))
            theta[base + 2 * k]     = np.log(2.0 * zeta * omega_n)   # log a1
            theta[base + 2 * k + 1] = np.log(omega_n ** 2)           # log a2
        # leave any trailing linear AR factor (index base + 2*num_quads_ar) as-is
        _ = has_linear_ar
    return theta


def _overdamped_ar_reset(warm_theta, data, p, q, rng,
                         c_span=0.2, zeta_range=(2.0, 6.0)):
    """Return a copy of ``warm_theta`` with the AR quadratic factors reset to an
    OVERDAMPED, low-frequency draw (the mirror of ``_underdamped_ar_reset``).

    Motivation (meeting 7/17 #9, the overdamped-init arm)
    -----------------------------------------------------
    The real Stripe 82 quasars are OVERDAMPED (zeta ~ 3.9: a slow relaxation pole
    tau ~ 95-125 d plus a fast pole tau ~ 1.8 d), the opposite corner from the sim's
    underdamped resonances. The optimizer's only extra AR-reset seed is the
    UNDERDAMPED one, so the overdamped basin has no dedicated initialization and can
    be under-reached from a warm (2,0) start. This seeds it: reset each AR quadratic
    to a low natural frequency (slow relaxation) with zeta > 1, keeping the two real
    poles inside the observable band, and let the optimizer descend. Off by default;
    enabled per run so it cannot alter the underdamped-sim behaviour.
    """
    theta = np.asarray(warm_theta, float).copy()
    d = data.d
    t_obs = np.asarray(data.t_obs, float)
    T_span = float(t_obs.max() - t_obs.min()) if t_obs.size > 1 else 1.0
    band = np.asarray(data.band)
    perband_dt = [np.median(np.diff(np.sort(t_obs[band == b])))
                  for b in np.unique(band) if int(np.sum(band == b)) > 1]
    dt_med = (float(max(perband_dt)) if perband_dt
              else (float(np.median(np.diff(np.sort(t_obs)))) if t_obs.size > 1 else 1.0))
    omega_n_min = 1.0 / (c_span * T_span) if T_span > 0 else 1e-3
    # keep the FAST overdamped pole omega_n*(zeta+sqrt(zeta^2-1)) below Nyquist:
    # bias omega_n to the slow end of the band (real overdamped poles are slow).
    omega_n_slow = omega_n_min * np.array([1.0, 5.0])       # a narrow slow band
    log_lo, log_hi = np.log(omega_n_slow[0]), np.log(omega_n_slow[1])

    num_quads_ar = p // 2
    for b in range(d):
        base = b * p
        for k in range(num_quads_ar):
            omega_n = float(np.exp(rng.uniform(log_lo, log_hi)))
            zeta = float(rng.uniform(*zeta_range))
            theta[base + 2 * k]     = np.log(2.0 * zeta * omega_n)   # log a1
            theta[base + 2 * k + 1] = np.log(omega_n ** 2)           # log a2
    return theta


def _ar_root_magnitudes(ar_factors):
    """Return the moduli ``|root|`` of all AR roots for one band.

    Uses the true (possibly complex) roots, so an under-damped quadratic factor
    contributes its natural frequency ``omega_n = |root| = sqrt(a2)`` rather than
    only the real part ``a1/2 = zeta*omega_n``.  This is what lets the
    observable-band / Nyquist check actually see the oscillation frequency and
    therefore catch aliasing of weakly-damped modes (small zeta, large omega_n),
    which the previous real-part-only magnitude could not.
    """
    mags = []
    for f in ar_factors:
        for r in _ar_roots_from_factor(f):
            mags.append(abs(r))
    return np.array(mags, dtype=float)


def _ma_root_magnitudes(ma_factors):
    """Return the moduli ``|root|`` of all MA roots for one band (true roots)."""
    mags = []
    for f in ma_factors:
        for r in _ma_roots_from_factor(f):
            mags.append(abs(r))
    return np.array(mags, dtype=float)


def _observational_identifiability(theta_carma, data, p, q,
                                   c_cadence=0.3, c_span=0.2):
    """Require observationally resolvable AR/MA time scales."""
    d = data.d
    ar_factors, ma_factors, _ = unpack_params_jones(theta_carma, d, p, q)

    if data.n <= 1:
        return True
    dt = np.diff(data.t_obs)
    dt_med = float(np.median(dt))
    T_span = float(data.t_obs.max() - data.t_obs.min())
    if dt_med <= 0.0 or T_span <= 0.0:
        return True

    # Observable-band walls (unchanged convention).  No Nyquist cap here:
    # capping omega_n at the Nyquist limit inside fit.py is the regression that
    # previously broke recovery of legitimately fast true modes (mu_b_diag_memo).
    # The aliasing fix is instead in _ar_root_magnitudes, which now measures the
    # root *modulus* |root| (= omega_n for an under-damped factor) rather than the
    # real part a1/2, so a weakly-damped aliased mode (|root| >> omega_max) is
    # caught by this existing wall.
    min_root_ma = 1.0 / (c_span * T_span)
    max_root_ma = 1.0 / (c_cadence * dt_med)
    min_root_ar = 1.0 / (c_span * T_span)
    max_root_ar = 1.0 / (c_cadence * dt_med)

    for band in range(d):
        for mag in _ar_root_magnitudes(ar_factors[band]):
            if mag < min_root_ar or mag > max_root_ar:
                return False
        for mag in _ma_root_magnitudes(ma_factors[band]):
            if mag < min_root_ma or mag > max_root_ma:
                return False

    return True


def _observational_identifiability_penalty(theta_carma, data, p, q,
                                            c_cadence=0.3, c_span=0.2,
                                            scale=1e8):
    """Compute a soft penalty for AR/MA time scales outside observable range."""
    d = data.d
    ar_factors, ma_factors, _ = unpack_params_jones(theta_carma, d, p, q)

    if data.n <= 1:
        return 0.0
    dt = np.diff(data.t_obs)
    dt_med = float(np.median(dt))
    T_span = float(data.t_obs.max() - data.t_obs.min())
    if dt_med <= 0.0 or T_span <= 0.0:
        return 0.0

    min_root_ma = 1.0 / (c_span * T_span)
    max_root_ma = 1.0 / (c_cadence * dt_med)
    min_root_ar = 1.0 / (c_span * T_span)
    max_root_ar = 1.0 / (c_cadence * dt_med)

    violation = 0.0
    for band in range(d):
        for mag in _ar_root_magnitudes(ar_factors[band]):
            if mag < min_root_ar:
                violation += (min_root_ar - mag) ** 2
            elif mag > max_root_ar:
                violation += (mag - max_root_ar) ** 2
        for mag in _ma_root_magnitudes(ma_factors[band]):
            if mag < min_root_ma:
                violation += (min_root_ma - mag) ** 2
            elif mag > max_root_ma:
                violation += (mag - max_root_ma) ** 2

    return scale * violation


def _band_prior_neglogp(theta_carma, data, p, q,
                        ar_lambda=0.0, ma_lambda=0.0,
                        c_cadence=0.3, c_span=0.2):
    """Soft, proper log-prior on AR/MA timescales (observable-band prior).

    This is the Bayesian replacement for the hard hinge in
    ``_observational_identifiability_penalty``. Instead of a near-hard wall
    (scale=1e8) in linear root-modulus space, it places a prior that is FLAT
    (zero ``-log p``) for every root modulus inside the observable frequency
    band ``[m_min, m_max]`` and rises as a quadratic in ``log`` frequency
    outside it:

        -log p = 0.5 * lambda * sum_roots [ h(log m_min - log m)^2
                                          + h(log m   - log m_max)^2 ],

    with ``h(x) = max(x, 0)``. Working in log-frequency makes the two walls
    symmetric and scale-free (a factor-of-10 excursion costs the same above
    and below the band). Because the prior is exactly zero in-band, it does
    not bias a resolvable timescale -- it only regularizes poles/zeros that
    try to leave the observable window. ``ar_lambda``/``ma_lambda`` are the
    per-family wall strengths (prior precisions); set to 0 to disable that
    family. The band edges ``m_min = 1/(c_span*T)``, ``m_max = 1/(c_cad*dt)``
    match the legacy hinge convention.
    """
    d = data.d
    ar_factors, ma_factors, _ = unpack_params_jones(theta_carma, d, p, q)

    if data.n <= 1:
        return 0.0
    # Resolvability is a PER-BAND property: each band sees ~1/d of the epochs, so
    # the GLOBAL interleaved median dt is ~d x finer and sets the observable
    # ceiling m_max = 1/(c_cadence*dt) ~d x too high. With the global dt the wall
    # only bites around z~5, leaving the whole per-band-UNobservable band (z in
    # (omega_max, ~5)) unpenalized -- exactly where the (2,1) fit parks the MA zero
    # (inactive + free), collapsing it below (2,0). Use the COARSEST per-band
    # cadence, matching the truth design (carma21_resolvability / run_task
    # dt_perband), so the wall sits at the true observable edge.
    band = np.asarray(data.band)
    t_obs = np.asarray(data.t_obs)
    perband_dt = [np.median(np.diff(np.sort(t_obs[band == b])))
                  for b in np.unique(band) if int(np.sum(band == b)) > 1]
    dt_med = float(max(perband_dt)) if perband_dt else float(np.median(np.diff(t_obs)))
    T_span = float(t_obs.max() - t_obs.min())
    if dt_med <= 0.0 or T_span <= 0.0:
        return 0.0

    log_min = np.log(1.0 / (c_span * T_span))
    log_max = np.log(1.0 / (c_cadence * dt_med))

    def _wall(mags):
        viol = 0.0
        for m in mags:
            if m <= 0.0:
                continue
            lm = np.log(m)
            if lm < log_min:
                viol += (log_min - lm) ** 2
            elif lm > log_max:
                viol += (lm - log_max) ** 2
        return viol

    neglogp = 0.0
    for band in range(d):
        if ar_lambda > 0.0:
            neglogp += 0.5 * ar_lambda * _wall(_ar_root_magnitudes(ar_factors[band]))
        if ma_lambda > 0.0:
            neglogp += 0.5 * ma_lambda * _wall(_ma_root_magnitudes(ma_factors[band]))
    return neglogp


def _damping_anticancel_neglogp(theta_carma, data, p, q,
                                damp_lambda=0.0, cancel_lambda=0.0,
                                c_cadence=0.3, c_span=0.2):
    """Soft log-priors closing the two Axis-A gaps of the observable-band prior
    (novus memo §8.3): a damping / marginal-stability FLOOR on AR poles, and a
    pole-zero ANTI-CANCELLATION separation prior.

    The band prior keys on |root| = omega_n, so it cannot see a pole driven
    toward the imaginary axis (zeta -> 0, Re lambda -> 0) whose omega_n is still
    in-band: a marginally-stable mode whose penalised likelihood has no interior
    optimum (it slides down a ridge toward the stability boundary -- the scipy
    Lyapunov "pole-pair-sum ~ 0" pathology, and exactly the failure seen on real
    object 2073759's (2,1) fit). For every AR pole lambda_k:

        -log p += 0.5 * damp_lambda * h(log delta_min - log|Re lambda_k|)^2,

    penalising damping |Re lambda_k| below ``delta_min`` -- the slowest decay
    rate resolvable in the window, 1/(c_span*T). h(x)=max(x,0), quadratic in log
    so it is scale-free and exactly zero for well-damped poles (no bias).

    Separately, nothing in the band prior stops an MA zero z_j from approaching
    an AR pole lambda_k (the (2,1)->(1,0) redundancy). For every (zero, pole)
    pair:

        -log p += 0.5 * cancel_lambda * h(log d_min - log|z_j - lambda_k|)^2,

    penalising complex-plane separations below ``d_min`` = 1/(c_span*T). Both
    families default to 0.0 (off), so existing callers are unchanged.
    """
    if damp_lambda <= 0.0 and cancel_lambda <= 0.0:
        return 0.0
    d = data.d
    ar_factors, ma_factors, _ = unpack_params_jones(theta_carma, d, p, q)
    if data.n <= 1:
        return 0.0
    dt = np.diff(data.t_obs)
    dt_med = float(np.median(dt))
    T_span = float(data.t_obs.max() - data.t_obs.min())
    if dt_med <= 0.0 or T_span <= 0.0:
        return 0.0

    # Slowest resolvable rate: poles that decay slower than this, or zero-pole
    # pairs closer than this, are observationally indistinguishable.
    log_floor = np.log(1.0 / (c_span * T_span))
    _TINY = 1e-12

    neglogp = 0.0
    for band in range(d):
        ar_roots = []
        for f in ar_factors[band]:
            ar_roots.extend(_ar_roots_from_factor(f))
        if damp_lambda > 0.0:
            for r in ar_roots:
                re = abs(r.real)
                lr = np.log(re if re > _TINY else _TINY)
                if lr < log_floor:
                    neglogp += 0.5 * damp_lambda * (log_floor - lr) ** 2
        if cancel_lambda > 0.0:
            ma_roots = []
            for f in ma_factors[band]:
                ma_roots.extend(_ma_roots_from_factor(f))
            for z in ma_roots:
                for lam in ar_roots:
                    sep = abs(z - lam)
                    ls = np.log(sep if sep > _TINY else _TINY)
                    if ls < log_floor:
                        neglogp += 0.5 * cancel_lambda * (log_floor - ls) ** 2
    return neglogp


def _band_pooling_neglogp(theta_carma, data, p, q,
                          ar_pool_lambda=0.0, ma_pool_lambda=0.0):
    """Partial-pooling (hierarchical) prior tying per-band AR/MA frequencies.

    The bands of one quasar share the same physical variability, so their
    characteristic frequencies should be similar. This prior shrinks each
    band's root frequencies toward a SHARED common value that is itself
    ESTIMATED from the data -- the hierarchical mean is profiled analytically:
    the per-family shared center is the cross-band mean of the current
    log-root-frequencies, and each band pays a quadratic cost for deviating
    from it,

        -log p = 0.5 * lambda * sum_k sum_b ( log m_{b,k} - mubar_k )^2,
        mubar_k = (1/d) sum_b log m_{b,k},

    where ``m_{b,k}`` is the k-th smallest AR (or MA) root modulus (= natural
    frequency ``omega_n``) of band ``b`` and ``d`` the band count. Working in
    log-frequency makes the shrinkage scale-free and couples the bands without
    pinning them to any fixed value -- partial, not complete, pooling, with
    ``lambda`` the pooling strength (prior precision). Profiling out ``mubar``
    adds no optimization parameters. With one band, ``lambda = 0``, or a
    ragged root count the term vanishes, so existing callers are unchanged.

    For AR the prior pools BOTH the natural frequency ``omega_n`` and the
    damping ratio ``zeta = a1/(2 sqrt(a0))`` of each quadratic factor (so bands
    are tied in both oscillation rate and decay), each under ``ar_pool_lambda``
    in log space; linear AR factors have no ``zeta`` and contribute frequency
    only. MA pools the zero frequencies under ``ma_pool_lambda``.
    """
    if ar_pool_lambda <= 0.0 and ma_pool_lambda <= 0.0:
        return 0.0
    d = data.d
    if d < 2:
        return 0.0
    ar_factors, ma_factors, _ = unpack_params_jones(theta_carma, d, p, q)
    _TINY = 1e-12

    def _pool(per_band_vals, lam, do_sort=True):
        logs = []
        for vals in per_band_vals:
            v = np.asarray(vals, dtype=float)
            if do_sort:
                v = np.sort(v)
            v = np.where(v > _TINY, v, _TINY)
            logs.append(np.log(v))
        L = np.vstack(logs)                       # (d, n_terms)
        center = L.mean(axis=0, keepdims=True)    # estimated shared mean per term
        return 0.5 * lam * float(np.sum((L - center) ** 2))

    def _ar_damping_ratios(factors):
        """zeta = a1/(2 sqrt(a0)) for each quadratic AR factor (s^2+a1 s+a0),
        ordered by natural frequency omega_n = sqrt(a0). Linear factors (no
        damping ratio) are skipped. Defined for over- and under-damped alike."""
        zw = []
        for f in factors:
            if len(f) == 2:
                a1, a0 = float(f[0]), float(f[1])
                wn = np.sqrt(a0) if a0 > 0.0 else _TINY
                zw.append((wn, a1 / (2.0 * wn)))
        zw.sort(key=lambda t: t[0])
        return np.array([z for _, z in zw], dtype=float)

    neglogp = 0.0
    if ar_pool_lambda > 0.0:
        ar_mags = [_ar_root_magnitudes(ar_factors[b]) for b in range(d)]
        n0 = len(ar_mags[0])
        if n0 > 0 and all(len(m) == n0 for m in ar_mags):
            neglogp += _pool(ar_mags, ar_pool_lambda)
        # Damping-ratio pooling (omega_n + zeta). zeta vectors are pre-ordered
        # by omega_n, so do NOT re-sort by value here.
        ar_zetas = [_ar_damping_ratios(ar_factors[b]) for b in range(d)]
        nz = len(ar_zetas[0])
        if nz > 0 and all(len(z) == nz for z in ar_zetas):
            neglogp += _pool(ar_zetas, ar_pool_lambda, do_sort=False)
    if ma_pool_lambda > 0.0 and q > 0:
        ma_mags = [_ma_root_magnitudes(ma_factors[b]) for b in range(d)]
        n0 = len(ma_mags[0])
        if n0 > 0 and all(len(m) == n0 for m in ma_mags):
            neglogp += _pool(ma_mags, ma_pool_lambda)
    return neglogp


def neg_loglik(theta, data, p, q, slopes=None,
               chol_ridge_lambda=0.0, chol_ridge_center=0.0,
               ar_band_lambda=None, ma_band_lambda=None,
               chol_ridge_onesided=False, chol_ridge_center_hi=None,
               ar_damping_lambda=None, pole_zero_lambda=None,
               ar_pool_lambda=None, ma_pool_lambda=None,
               corr_ridge_lambda=0.0, diag_load_lambda=0.0):
    """
    Evaluate the negative Gaussian innovations log-likelihood for an
    MCARMA(p, q) model.

    Parameters
    ----------
    theta : ndarray
        Unconstrained optimisation vector containing

            [AR parameters,
             MA parameters,
             vech(log-Cholesky process covariance),
             band means]

        under the Jones-style log parameterization implemented in
        `unpack_params_jones()`.

        Positive polynomial coefficients and Cholesky diagonal entries are
        represented in log-space so that optimisation is unconstrained in
        R^n while preserving stability-compatible coefficient signs after
        exponentiation.

    data : ObservationData
        Irregularly sampled multiband observations.

    p, q : int
        AR and MA orders of the MCARMA model.

    slopes : ndarray (d,) or None, default None
        Per-band slopes of a known/fixed deterministic linear trend in
        observation time, passed through to `kalman.loglik`. When None,
        no trend is modelled (default, unchanged behaviour).

    chol_ridge_lambda : float, default 0.0
        Strength of an optional Gaussian ridge / prior on the Cholesky
        log-variances (the diagonal entries of the vech block, which equal
        ``log Sigma_bb``). When > 0 a term

            ``0.5 * chol_ridge_lambda * sum_b (theta_chol_bb - center)**2``

        is ADDED to the returned negative log-likelihood. Its purpose is to
        keep a process-variance component from diverging to -inf (Sigma_bb ->
        0, a boundary solution at the edge of the positive-definite cone) so
        the optimum stays interior and the Hessian is computable. Default 0.0
        leaves behaviour unchanged.

    chol_ridge_center : float or array_like (d,), default 0.0
        Target log-variance the Gaussian Sigma-prior pulls toward (units of
        ``log Sigma_bb``). A scalar applies the same center to every band; a
        length-``d`` vector supplies a per-band center (e.g. the data-driven
        ``2*log(0.1*sigma_hat_obs_b)``). Only used when
        ``chol_ridge_lambda > 0``. Together with ``chol_ridge_lambda`` this term
        is a log-normal prior on the process variances, ``log Sigma_bb ~
        N(center, 1/chol_ridge_lambda)``.

    chol_ridge_onesided : bool, default False
        When False (default) the Sigma-prior is the symmetric Gaussian above.
        When True it becomes a one-sided soft FLOOR: only log Sigma_bb values
        BELOW ``center`` are penalised, via ``0.5 * chol_ridge_lambda *
        sum_b max(center_b - log Sigma_bb, 0)**2`` (a half-normal prior). This
        leaves well-identified / large variances exactly unbiased and pushes
        only collapsing variances (those the data cannot distinguish from the
        PD-cone boundary) up to the floor, where the prior supplies the
        curvature that makes the Hessian PD.

    chol_ridge_center_hi : array_like (d,) or None, default None
        Optional upper center (units of ``log Sigma_bb``) that turns the
        one-sided floor into a two-sided soft BOX. Only used when
        ``chol_ridge_onesided`` is True. When supplied, log Sigma_bb values
        ABOVE ``chol_ridge_center_hi`` are penalised by the same
        ``0.5 * chol_ridge_lambda * sum_b max(log Sigma_bb - center_hi_b, 0)**2``
        half-normal wall. The prior is flat between ``chol_ridge_center`` and
        ``chol_ridge_center_hi`` (no bias on physically plausible variances) and
        walls off the ``Sigma -> inf`` runaway that the Sigma<->AR-damping
        trade-off opens when the floor alone leaves the upper side unbounded.
        ``None`` (default) keeps the pure one-sided floor.

    ar_band_lambda, ma_band_lambda : float or None, default None
        Strengths (prior precisions) of the soft observable-band log-prior on
        the AR / MA root frequencies (see ``_band_prior_neglogp``). When BOTH
        are None the legacy near-hard hinge
        (``_observational_identifiability_penalty``, scale=1e8) is used,
        preserving previous behaviour for existing callers. When either is set
        (float, may be 0 to disable just that family) the soft log-frequency
        prior is used INSTEAD of the hinge, so the two are never double-counted.

    ar_damping_lambda, pole_zero_lambda : float or None, default None
        Strengths of the Axis-A §8.3 priors that close the two gaps of the
        observable-band prior (see ``_damping_anticancel_neglogp``).
        ``ar_damping_lambda`` is a one-sided log-floor on AR-pole damping
        ``|Re lambda_k|`` (penalising poles driven toward the imaginary axis /
        marginal stability, where the penalised likelihood otherwise has no
        interior optimum). ``pole_zero_lambda`` penalises an MA zero approaching
        an AR pole (the (2,1)->(1,0) redundancy). Both default to None/0 = off,
        added on top of whichever band prior is active.

    Returns
    -------
    float
        Negative log-likelihood. Invalid parameterizations, unstable state
        matrices, or Kalman filter failures return a large penalty value.

    Notes
    -----
    The likelihood is computed from the continuous-time state-space
    realization returned by `build_state_space()` and evaluated with the
    Kalman innovations filter implemented in `kalman.loglik()`.

    Stability is enforced by requiring all eigenvalues of the continuous-time
    drift matrix F to lie strictly in the left half-plane.
    """
    try:
        d           = data.d
        mu_vec      = theta[-d:].reshape(d, 1)
        theta_carma = theta[:-d]
        ar_factors, ma_factors, Sigma = unpack_params_jones(theta_carma, d, p, q)

        F, G, H, Sigma = build_state_space(ar_factors, ma_factors, Sigma, d, p, q)

        # Optional diagonal loading (Zhirui 2026-08-13): a MODEL regularizer (not a
        # penalty). Replace the driving covariance with Sigma + lam*mean(diag)*I
        # inside the likelihood, lifting the smallest eigenvalue off the near-
        # singular cross-band boundary. Relative (mean-diagonal) scaling keeps
        # diag_load_lambda dimensionless / cell-scale-free. Default 0 = off =
        # byte-identical. Mirrors jax_loglik._state_space; the numpy-vs-jax parity
        # test pins the two together.
        if diag_load_lambda and diag_load_lambda > 0.0:
            Sigma = Sigma + diag_load_lambda * (np.trace(Sigma) / d) * np.eye(d)

        lambda_max_real = np.max(np.real(np.linalg.eigvals(F)))
        if lambda_max_real >= -1e-4:
            return 1e10

        ll = loglik(data.t_obs, data.y_obs, F, G, H, Sigma,
                    data.C_list, data.R_list, mu_vec,
                    slopes=slopes,
                    lambda_max_real=lambda_max_real)
        if not np.isfinite(ll):
            return 1e10

        # AR/MA observable-band prior. Default (both None) keeps the legacy
        # near-hard hinge; otherwise use the soft log-frequency prior instead.
        if ar_band_lambda is None and ma_band_lambda is None:
            penalty = _observational_identifiability_penalty(
                theta_carma, data, p, q
            )
        else:
            penalty = _band_prior_neglogp(
                theta_carma, data, p, q,
                ar_lambda=(ar_band_lambda or 0.0),
                ma_lambda=(ma_band_lambda or 0.0),
                c_cadence=0.7, c_span=0.2,
            )

        # Axis-A §8.3 priors: AR-pole damping floor + MA/AR pole-zero
        # anti-cancellation. Added on top of whichever band prior is active;
        # both default off (None/0) so behaviour is unchanged for old callers.
        if ar_damping_lambda or pole_zero_lambda:
            penalty += _damping_anticancel_neglogp(
                theta_carma, data, p, q,
                damp_lambda=(ar_damping_lambda or 0.0),
                cancel_lambda=(pole_zero_lambda or 0.0),
            )

        # Partial-pooling (hierarchical) prior coupling per-band AR/MA
        # frequencies toward an estimated shared mean. Both default off
        # (None/0) so existing callers are unchanged.
        if ar_pool_lambda or ma_pool_lambda:
            penalty += _band_pooling_neglogp(
                theta_carma, data, p, q,
                ar_pool_lambda=(ar_pool_lambda or 0.0),
                ma_pool_lambda=(ma_pool_lambda or 0.0),
            )

        ridge = 0.0
        if chol_ridge_lambda > 0.0:
            # Cholesky block is theta_carma[n_ar + n_ma:]; its diagonal entries
            # (tril positions with i == j) are packed as theta = 2*log L_bb =
            # log(L_bb^2) -- i.e. they ARE the log-variance already (see
            # unpack_params_jones: L_ii = exp(0.5*theta); reporting packer stores
            # 2*log L_ii). The Sigma-prior centers (chol_ridge_center[_hi]) are in
            # the same log-variance units (data-driven 2*log(floor_mult*sd)), so
            # logvar = diag directly.
            # NOTE (2026-07 bug fix): this was `logvar = 2.0 * diag`, which
            # double-counted the already-doubled param (-> 4*log L_bb) and made the
            # nominal 1e-4*sd floor bind ~100x too high in L_bb, clamping
            # well-identified small trailing-Cholesky diagonals at high rho and
            # collapsing multivariate coverage (0.96 -> 0.30). See memory
            # mv-coverage-fix-options / the sigma-floor-factor2-bug.
            n_ar = d * p
            n_ma = d * q
            chol_block = theta_carma[n_ar + n_ma:]
            rows, cols = np.tril_indices(d)
            diag = chol_block[rows == cols]
            logvar = diag                            # log Sigma_bb (param is 2*log L_bb)
            center = np.asarray(chol_ridge_center, dtype=float)
            if chol_ridge_onesided:
                # One-sided soft FLOOR: penalise only log Sigma_bb BELOW the
                # center (variance collapsing toward the PD-cone boundary).
                # h(center - logvar) = max(center - logvar, 0) is exactly zero
                # for well-identified / large variances, so they are left
                # unbiased; only directions the data cannot pin down get lifted
                # to the floor. A one-sided (half-normal) prior on log Sigma_bb.
                dev = np.maximum(center - logvar, 0.0)
                ridge = 0.5 * chol_ridge_lambda * np.sum(dev ** 2)
                if chol_ridge_center_hi is not None:
                    # Optional soft CEILING -> two-sided soft BOX. Penalise only
                    # log Sigma_bb ABOVE chol_ridge_center_hi. Between the floor
                    # and ceiling the prior is flat (no bias on physically
                    # plausible variances); it walls off only the Sigma -> inf
                    # runaway that the Sigma<->AR-damping trade-off opens up when
                    # the floor alone leaves the upper side unbounded.
                    center_hi = np.asarray(chol_ridge_center_hi, dtype=float)
                    dev_hi = np.maximum(logvar - center_hi, 0.0)
                    ridge += 0.5 * chol_ridge_lambda * np.sum(dev_hi ** 2)
            else:
                # Symmetric Gaussian prior on log Sigma_bb (legacy shape).
                dev = logvar - center
                ridge = 0.5 * chol_ridge_lambda * np.sum(dev ** 2)

        # Cross-band correlation ridge (Zhirui 2026-08-11). Shrinks the OFF-
        # diagonal correlations of the innovation covariance toward zero, i.e.
        # pulls the fitted Sigma away from the near-singular cross-band corner
        # that carries the +0.02..+0.03 rho over-estimation. This is the only
        # penalty that touches correlations; sigma_softbox/chol_ridge constrain
        # marginal variances only. Default off (lambda 0) so production and every
        # existing caller are byte-identical; a diagnostic refit arm, not
        # production. Penalty = 0.5 * lambda * sum_{b<c} R_bc^2 on the correlation
        # R = diag(Sigma)^-1/2 Sigma diag(Sigma)^-1/2.
        if corr_ridge_lambda > 0.0:
            sd = np.sqrt(np.clip(np.diag(Sigma), 1e-300, None))
            R = Sigma / np.outer(sd, sd)
            iu = np.triu_indices(d, 1)
            ridge = ridge + 0.5 * corr_ridge_lambda * np.sum(R[iu] ** 2)

        return -ll + penalty + ridge
        # return -ll
    except Exception:
        return 1e10


def _pso_search(obj, x0, delta, n_particles=40, iters=200, seed=0,
                w=0.72, c1=1.49, c2=1.49, topology="global"):
    """Derivative-free particle-swarm minimiser of ``obj`` inside the box
    ``x0 +/- delta`` (per-coordinate half-width), with particle 0 seeded at the
    incumbent ``x0`` so the swarm's best is guaranteed no worse than ``x0``.

    Used by ``fit(pso_polish=True)`` to globally refine the local optimum. This
    is the honest version of the deprecated overdamped-init arm (meeting 7/17
    #7): it explores basin structure with no gradients, and -- unlike that arm
    -- does NOT drive Sigma->0 (the standalone 7/17 diagnostic confirmed PSO
    closes the real (2,1) loglik gap without a Sigma collapse). Pure numpy.

    Swarm hyper-parameters (meeting 7/23 #3, tuning toward EXPLORATION):
      w   -- inertia weight (higher = more exploration / momentum).
      c1  -- cognitive pull to each particle's own best (higher = more
             independent exploration).
      c2  -- social pull to the neighbourhood best (lower = slower collapse
             onto the incumbent basin, less premature convergence).
      topology -- neighbourhood the social term follows:
        "global" : fully-connected star (gbest); fastest, most exploitative.
        "ring"   : each particle sees only neighbours i-1,i,i+1 (lbest); slows
                   information flow so the swarm can hold several basins at once
                   and is more likely to migrate a local optimum to the global.

    Returns ``(best_x, best_obj)``.
    """
    rng = np.random.default_rng(seed)
    x0 = np.asarray(x0, float)
    dim = len(x0)
    lb, ub = x0 - delta, x0 + delta
    X = rng.uniform(lb, ub, (n_particles, dim))
    X[0] = x0                                     # seed incumbent (PSO >= x0)
    V = rng.uniform(-1.0, 1.0, (n_particles, dim)) * delta * 0.1
    pbest = X.copy()
    pbest_f = np.array([obj(x) for x in X])
    g = int(np.argmin(pbest_f))
    gbest, gbest_f = pbest[g].copy(), float(pbest_f[g])

    if topology == "ring":
        # neighbour index sets {i-1, i, i+1} (mod n) for the lbest social term
        nb = [((i - 1) % n_particles, i, (i + 1) % n_particles)
              for i in range(n_particles)]

    def _social_targets():
        if topology == "ring":
            tgt = np.empty_like(X)
            for i in range(n_particles):
                j = min(nb[i], key=lambda k: pbest_f[k])   # best neighbour
                tgt[i] = pbest[j]
            return tgt
        return gbest                                        # broadcast (global)

    for _ in range(int(iters)):
        r1 = rng.random((n_particles, dim)); r2 = rng.random((n_particles, dim))
        social = _social_targets()
        V = w * V + c1 * r1 * (pbest - X) + c2 * r2 * (social - X)
        X = np.clip(X + V, lb, ub)
        f = np.array([obj(x) for x in X])
        imp = f < pbest_f
        pbest[imp], pbest_f[imp] = X[imp], f[imp]
        g = int(np.argmin(pbest_f))
        if pbest_f[g] < gbest_f:
            gbest, gbest_f = pbest[g].copy(), float(pbest_f[g])
    return gbest, gbest_f


SIGMA_COND_MAX = 1e12
"""Condition number of Sigma above which a fit is on the Sigma->0 boundary.

Measured, not chosen. The degeneracy census over the LSST bright arm (712
records, 2136 mv fits) splits cleanly at this value: of the 818 fits with an
interior Sigma only 8 exceed it, while of the 613 collapsed fits 611 do. The
two populations barely overlap, and the separation survives the change of
parameterization that a threshold on the Cholesky log-diagonal does not.
"""


def sigma_condition(theta, d, p, q):
    """Condition number of the driving covariance encoded in a full theta.

    ``theta`` is the packed vector fit() optimizes, mu included; the trailing d
    mean components are dropped before unpacking. Returns inf when Sigma is
    singular or theta cannot be unpacked, so callers can treat "unusable" and
    "maximally ill conditioned" the same way.
    """
    try:
        _, _, Sig = unpack_params_jones(np.asarray(theta, float)[:-d], d, p, q)
        ev = np.linalg.eigvalsh(np.asarray(Sig, float))
        emin, emax = float(ev.min()), float(ev.max())
        return emax / emin if emin > 0.0 else np.inf
    except Exception:
        return np.inf


def restart_quality_tier(theta, score, d, p, q, cond_max=None, grad=None,
                         gtol_rel=1e-4):
    """Rank band for a scorable restart. Higher is better.

    3  stationary and well conditioned -- an interior maximum
    2  well conditioned, not stationary -- interior but unfinished
    1  ill conditioned -- on the Sigma->0 boundary, whatever its gradient
    0  unscorable (set by fit(), never returned here)

    The ORDER matters and is not the obvious one. Stationarity on its own is
    worse than useless as a rank key: the likelihood asymptotes as Sigma_bb->0,
    so boundary points have a vanishing gradient and pass the stationarity test
    MORE often than interior ones (75.3% vs 72.8% in the census). Tiering on it
    alone would promote the degenerate fits over the salvageable ones, which is
    the opposite of the intent. Conditioning has to gate it, and an unfinished
    interior search outranks a boundary point because it is the one that more
    polishing can still rescue.

    Both gates are opt-in. With cond_max and grad left at None every restart is
    tier 1, which is the historical ranking exactly, so the arms fitted to date
    are unaffected and remain reproducible.
    """
    tier = 1
    if cond_max is not None:
        if sigma_condition(theta, d, p, q) > float(cond_max):
            return 1
        tier = 2
    if grad is not None and tier == 2:
        try:
            gn = float(grad(np.asarray(theta, float)))
        except Exception:
            return tier
        if np.isfinite(gn) and gn <= gtol_rel * max(1.0, abs(score)):
            tier = 3
    return tier


PRECOND_CLAMP = 1e3
"""Widest departure from the median step scale the preconditioner may apply."""


def curvature_scales(hess_diag, clamp=PRECOND_CLAMP):
    """Per-parameter step scales that unstretch the likelihood surface.

    BFGS starts from an identity inverse Hessian, so it implicitly assumes every
    parameter wants the same step, and it has to learn the real shape from the
    steps it takes. The MCARMA likelihood is a bad match for that assumption:
    the curvature diagonal spans roughly twelve decades on an interior fit, so
    most of the iteration budget goes into learning the scaling rather than
    descending. That is consistent with what the arm shows -- the large majority
    of non-converged fits stop at maxiter, not at a stationary point. Optimizing
    in z with ``theta = scales * z`` makes the curvature in z roughly uniform,
    which is the regime BFGS is actually efficient in. Measured effect on 41
    real fits: condition number of the interior Hessians falls from about
    10^12.8 to 10^6.5.

    ``scales_i = |H_ii|^(-1/2)``, normalized to unit median and clipped to
    ``[1/clamp, clamp]``. The normalization matters: it keeps the gradient norm
    in z comparable to the one in theta, so ``gtol`` keeps its meaning and the
    downstream acceptance test (which runs in theta on the plain likelihood) is
    completely untouched by this change. The clip bounds how much damage a
    preconditioner built at a bad reference point can do, and at the default it
    still spans the six decades of step scale that twelve decades of curvature
    imply, so it is not binding on a healthy fit.

    A flat, zero or non-finite entry falls back to the MEDIAN curvature rather
    than to an enormous step, so a collapsing Sigma block cannot hand one
    coordinate a scale that swamps every other. If nothing is usable at all the
    result is all ones, which is exactly the un-preconditioned search.
    """
    h = np.abs(np.asarray(hess_diag, float)).ravel()
    good = np.isfinite(h) & (h > 0.0)
    if not good.any():
        return np.ones(h.size)
    h = np.where(good, h, np.median(h[good]))
    s = 1.0 / np.sqrt(h)
    s = s / np.median(s)
    return np.clip(s, 1.0 / float(clamp), float(clamp))


def curvature_scales_at(data, p, q, theta, slopes=None, clamp=PRECOND_CLAMP,
                        **prior):
    """Build curvature scales from the exact Hessian diagonal at ``theta``.

    Split out of fit() so a caller running SEVERAL searches from one point (the
    stage-2 ladder, which walks up to four rungs from the same warm start) pays
    for the Hessian once rather than once per rung. The Hessian costs n_theta
    forward-over-reverse passes, each about as expensive as one gradient, so on
    a long DDF curve it is not free.

    Raises whatever jax_loglik raises; callers decide whether an unavailable
    preconditioner is fatal (it should not be: the un-preconditioned search is
    what every fit to date already ran).
    """
    from .jax_loglik import make_scipy_hessian_hvp
    hess_fn = make_scipy_hessian_hvp(data, p, q, slopes=slopes, **prior)
    return curvature_scales(np.diag(hess_fn(np.asarray(theta, float))),
                            clamp=clamp)


def precondition_objective(fun_and_grad, scales):
    """Wrap a value+grad objective so the optimizer searches in scaled coords.

    ``g(z) = f(scales * z)`` and ``grad g(z) = scales * grad f(scales * z)``.
    """
    s = np.asarray(scales, float)

    def g(z):
        val, grad = fun_and_grad(s * np.asarray(z, float))
        return val, s * np.asarray(grad, float)

    return g


def unscale_result(res, scales):
    """Map a scipy result back out of scaled coordinates into theta space.

    The point maps as ``theta = scales * z`` and the gradient as
    ``grad f = grad g / scales``. The inverse Hessian maps as
    ``H_theta^-1 = diag(s) H_z^-1 diag(s)``, and that one is load bearing: the
    BFGS inverse Hessian is what the ``hess_inv_diag`` standard errors are read
    off. Skipping it would leave every fitted value correct and silently rescale
    every SE, so it is handled here rather than left to the caller to remember.
    """
    s = np.asarray(scales, float)
    res.x = s * np.asarray(res.x, float)
    jac = getattr(res, "jac", None)
    if isinstance(jac, np.ndarray) and jac.shape == s.shape:
        res.jac = np.asarray(jac, float) / s
    hi = getattr(res, "hess_inv", None)
    if hi is not None and not isinstance(hi, np.ndarray) and hasattr(hi, "todense"):
        try:
            hi = np.asarray(hi.todense())
        except Exception:
            hi = None
    if isinstance(hi, np.ndarray) and hi.shape == (s.size, s.size):
        res.hess_inv = s[:, None] * hi * s[None, :]
    return res


def fit(data, p, q=0, n_restarts=20, method="BFGS",
        warm_theta=None, seed=None, use_dual_annealing=False,
        no_local_search=False, final_polish=False, slopes=None,
        chol_ridge_lambda=0.0, chol_ridge_center=0.0,
        ar_band_lambda=None, ma_band_lambda=None,
        chol_ridge_onesided=False, chol_ridge_center_hi=None,
        ar_damping_lambda=None, pole_zero_lambda=None,
        ar_pool_lambda=None, ma_pool_lambda=None,
        corr_ridge_lambda=0.0, diag_load_lambda=0.0,
        ma_above_ar=False, maxiter=1000,
        use_jax_grad=False, jax_gtol=1e-5, overdamped_reset=False,
        pso_polish=False, pso_particles=40, pso_iters=200,
        random_start_damping="under", local_only=False,
        pso_w=0.72, pso_c1=1.49, pso_c2=1.49, pso_topology="global",
        restart_score=None, restart_cond_max=None, restart_grad=None,
        restart_gtol_rel=1e-4, precondition=None):
    """
    Fit MCARMA(p, q) by maximum likelihood with multiple random restarts.

    Uses unbounded BFGS locally (method="BFGS") for cleaner unconstrained
    optimization in log-parameter space.

    slopes : ndarray (d,) or None, default None
        Per-band slopes of a known/fixed deterministic linear trend in
        observation time. When supplied, the trend ``slopes[band] * t`` is
        modelled inside the likelihood (added to the predicted mean) rather
        than being removed from the data beforehand. When None (default), no
        trend is modelled and behaviour is unchanged. The value is forwarded
        to `neg_loglik`/`kalman.loglik` and echoed back in the result dict.

    Warm start behaviour:
      - Restart 1 uses warm_theta exactly (no jitter).
      - Restarts 2 through n_restarts//2 add small Gaussian jitter
        (scale=1e-3) around warm_theta to explore the local basin.
      - Remaining restarts draw fully random initializations.

    precondition : {"curvature", None}, default None
        Reparameterize the search as ``theta = scales * z`` before optimizing,
        with ``scales`` built from the exact Hessian diagonal at the warm start
        (see curvature_scales). Requires use_jax_grad, since the scales come
        from the same penalized objective the search minimizes; silently falls
        back to the raw search if the Hessian cannot be built. Default None
        reproduces every fit run to date exactly. Nothing outside the search
        changes: the returned theta, the acceptance test and the standard errors
        are all mapped back into theta space before _record sees them.

    local_only : bool, default False
        With a valid warm_theta, search from that point and nowhere else: no
        jitter, no random restarts, and none of the damping-reset seeds that are
        otherwise appended for p>=2. The result is then a pure local descent, so
        the returned objective cannot exceed the objective at warm_theta. Used
        by the two-stage MLE finish, where an ascent that is guaranteed relative
        to its own starting point is the whole point of the stage.

    restart_score : callable or None, default None
        Function of theta, higher is better, used to RANK the restarts against
        each other. None keeps the historical behaviour: rank on the objective
        each restart minimized, which is what every arm to date was fitted with.

        Supply it when the search minimizes something other than the function
        the result will be reported on, which is exactly the two-stage finish.
        Two ways that happens here. Under use_jax_grad the restarts minimize the
        JAX objective while the estimate is reported on the Kalman likelihood;
        the two agree to six decimals at a sane theta and can differ by hundreds
        of nats at a runaway MA coefficient. And for p>=2 the seed list gets
        damping resets that land in other basins. So the restart with the best
        res.fun need not be the one with the best reported likelihood, and every
        losing restart is otherwise discarded without ever being scored on it.

    restart_cond_max : float or None, default None
        Condition number of Sigma above which a restart is treated as sitting on
        the Sigma->0 boundary and demoted below every well conditioned restart,
        however good its likelihood. 1e12 is the measured separator; see
        _quality_tier. None (default) disables the gate and the ranking is
        exactly the historical one.

    restart_grad : callable or None, default None
        Function of theta returning the infinity norm of the gradient of the
        REPORTED likelihood. Used only to promote a well conditioned restart
        that has actually converged above one that has not. Ignored unless
        restart_cond_max is also set, because a vanishing gradient at the
        boundary means degeneracy rather than convergence and promoting on it
        alone inverts the ranking. None (default) disables the promotion.

    restart_gtol_rel : float, default 1e-4
        Relative tolerance for that stationarity promotion: a restart counts as
        converged when ||grad||inf <= restart_gtol_rel * max(1, |score|), the
        same test stage2.finish_mle accepts on.

        A restart whose score is non-finite is DEMOTED rather than dropped: it
        loses to any scorable restart and comes back only when no restart was
        scorable at all. So an unscorable point can never displace a scorable
        one, while a fit whose restarts are every one unscorable still returns a
        point instead of raising. The returned dict records which metric decided
        the winner under result["restarts"]["ranked_by"] --
        "objective_fallback" marks that last case -- and how many restarts were
        demoted under result["restarts"]["unscorable"].

    Note: near-zero innovation variance is a likelihood-geometry issue
    (pole-zero near-cancellation, covariance-degenerate basin) rather than
    a parameterization failure. Positivity is guaranteed by the log-Cholesky
    parameterization. Diagnosis should use diag_stationary.py profile scans.
    """
    d       = data.d
    n_ar    = d * p
    n_ma    = d * q
    n_chol  = d * (d + 1) // 2
    n_theta = n_ar + n_ma + n_chol + d

    bounds = ([(None, None)] * n_ar
            + [(None, None)] * n_ma
            + [(None, None)] * n_chol
            + [(None, None)] * d)

    best_ll     = -np.inf
    # The value restarts are ranked against. It IS best_ll unless restart_score
    # is given, in which case the two come apart and best_ll stays the objective
    # value purely for the log line and the provenance block.
    best_score  = -np.inf
    # Ranking tier, so a restart that restart_score cannot score can still be
    # returned as a last resort without ever outranking one it can. Tier 1 is
    # scored on restart_score (on the objective when no restart_score is given);
    # tier 0 is a restart whose reported likelihood is undefined, ranked among
    # its own kind on the objective. Dropping tier-0 restarts outright instead
    # makes fit() raise "All optimisation restarts failed" whenever EVERY
    # restart lands where the reported likelihood is undefined, which is a real
    # case and not a pathological one: an optimum of the LOADED model can sit at
    # a Sigma that is singular unloaded, so the plain likelihood is nan there.
    # Raising loses the point entirely, and a caller that runs its stage-2
    # attempt inline rather than through finish_mle (fit_lsst_downsample.fit_one)
    # skips its whole escalation-and-fallback path on the exception and reports
    # the loaded stage-1 fit as if it were unloaded. Returning the point lets the
    # caller's own acceptance test reject it on the merits, which is what the
    # escalation ladder is for.
    best_tier    = -1
    n_unscorable = 0
    best_result = None
    rng         = np.random.default_rng(seed)

    # Restart accounting (meeting item 11, 2026-08-19). Under the two-stage
    # estimator the penalized search and the plain-likelihood finish are two
    # separate calls into fit(), and a pooled restart count cannot say whether a
    # stage-2 success=False is a stage-2 failure or a stage-1 handoff that was
    # never good enough. Each call therefore reports its own budget, how many
    # restarts produced a finite objective, and which one won.
    n_attempted = 0
    n_finite    = 0
    best_label  = None
    best_nit    = -1
    best_nfev   = -1

    # Extra args appended to every neg_loglik call so the optional priors
    # (Gaussian Sigma-prior + soft AR/MA band prior) are applied consistently
    # across all optimiser paths. Order matches neg_loglik's positional tail:
    # (chol_ridge_lambda, chol_ridge_center, ar_band_lambda, ma_band_lambda,
    # chol_ridge_onesided, chol_ridge_center_hi, ar_damping_lambda,
    # pole_zero_lambda, ar_pool_lambda, ma_pool_lambda, corr_ridge_lambda,
    # diag_load_lambda).
    ridge_args = (chol_ridge_lambda, chol_ridge_center,
                  ar_band_lambda, ma_band_lambda, chol_ridge_onesided,
                  chol_ridge_center_hi, ar_damping_lambda, pole_zero_lambda,
                  ar_pool_lambda, ma_pool_lambda, corr_ridge_lambda,
                  diag_load_lambda)

    def _quality_tier(x, score):
        return restart_quality_tier(
            x, score, d, p, q,
            cond_max=restart_cond_max, grad=restart_grad,
            gtol_rel=restart_gtol_rel,
        )

    def _record(res, label):
        nonlocal best_ll, best_score, best_result, best_tier, n_unscorable
        nonlocal n_attempted, n_finite, best_label, best_nit, best_nfev

        n_attempted += 1
        if not np.isfinite(res.fun):
            _log.info(f"[fit] {label}: non-finite, skipping")
            return
        n_finite += 1

        ll = -res.fun
        nit = getattr(res, "nit", -1)
        nfev = getattr(res, "nfev", -1)

        # Rank on the reported function when the caller supplies one, on the
        # minimized objective otherwise. See the restart_score docstring: these
        # are not the same ordering whenever the search and the report disagree,
        # and only the winner ever gets scored on the reported function, so a
        # better point found in a losing restart is lost unless it is ranked
        # here.
        score = ll
        tier  = 1
        if restart_score is not None:
            try:
                score = float(restart_score(np.asarray(res.x, float)))
            except Exception:
                score = float("nan")
            if not np.isfinite(score):
                # Demoted, not dropped: it can only be returned if NO restart
                # was scorable, and the caller's acceptance test still has to
                # pass on it before anything is reported.
                n_unscorable += 1
                tier, score = 0, ll
                _log.info(f"[fit] {label}: obj={ll:.4f} but unscorable on the "
                      f"reported likelihood, demoted")
            else:
                tier = _quality_tier(res.x, score)
                _log.info(f"[fit] {label}: obj={ll:.4f} score={score:.4f} "
                      f"tier={tier}, best={max(score, best_score):.4f} "
                      f"(nit={nit} nfev={nfev} "
                      f"success={getattr(res, 'success', '?')})")
        else:
            _log.info(f"[fit] {label}: ll={ll:.4f}, best={max(ll, best_ll):.4f} "
                  f"(nit={nit} nfev={nfev} success={getattr(res, 'success', '?')})")

        if (tier, score) > (best_tier, best_score):
            best_label, best_nit, best_nfev = label, int(nit), int(nfev)
            try:
                mu_fitted   = res.x[-d:].reshape(d, 1)
                theta_carma = res.x[:-d]

                ar_f, ma_f, Sig = unpack_params_jones(
                    theta_carma, d, p, q
                )

                F_, G_, H_, Sig = build_state_space(
                    ar_f, ma_f, Sig, d, p, q
                )

                # Carry the diagonal loading into the REPORTED state space,
                # mirroring neg_loglik (which loads after build_state_space).
                # Without this the objective maximizes the loaded model while
                # `Sigma`, `loglik_pure` and every downstream consumer (AICc,
                # RTS smoother, PSD, GOF) describe the UNLOADED one, so the
                # reported likelihood is neither model's maximum -- an
                # order-dependent shortfall of exactly the kind two-stage
                # estimation exists to remove. This keeps a single fit() call
                # self-consistent; it is NOT a claim about what AICc should
                # score. Under the two-stage estimator loading is applied in
                # the penalized stage only and dropped for the plain-MLE finish
                # (user directive 2026-08-20), so the scored optimum is the
                # unloaded one. See fit_extra_s2 in fits/21sim/sim_study.py.
                if diag_load_lambda and diag_load_lambda > 0.0:
                    Sig = Sig + diag_load_lambda * (np.trace(Sig) / d) * np.eye(d)

            except Exception:
                return

            # ----------------------------------------------------------
            # Extract optimizer inverse Hessian approximation
            #
            # BFGS:
            #   ndarray
            #
            # L-BFGS-B:
            #   LbfgsInvHessProduct -> convert via todense()
            # ----------------------------------------------------------
            hess_inv = getattr(res, "hess_inv", None)

            try:
                if hess_inv is not None and not isinstance(hess_inv, np.ndarray):
                    if hasattr(hess_inv, "todense"):
                        hess_inv = np.asarray(hess_inv.todense())
                    else:
                        hess_inv = None
            except Exception:
                hess_inv = None

            best_ll = ll
            best_score = score
            best_tier = tier

            # PURE (unpenalized) Kalman log-likelihood at the fitted theta.
            # `ll = -res.fun` above is the *penalized* objective value
            # (pure_ll - penalty - ridge); AIC/AICc must be scored on the pure
            # maximized log-likelihood, not the penalized objective, so we
            # recompute the likelihood term here without any penalty. The fit
            # itself (and the SE Hessian, which is the penalized Hessian) uses the
            # penalized objective. Callers should feed `loglik_pure` to
            # compute_aicc; `loglik` is retained for provenance/back-compat.
            try:
                _lmax = float(np.max(np.real(np.linalg.eigvals(F_))))
                loglik_pure = float(loglik(
                    data.t_obs, data.y_obs, F_, G_, H_, Sig,
                    data.C_list, data.R_list, mu_fitted,
                    slopes=slopes, lambda_max_real=_lmax,
                ))
            except Exception:
                loglik_pure = ll

            best_result = {
                "F": F_,
                "G": G_,
                "H": H_,
                "Sigma": Sig,
                "mu": mu_fitted,
                "theta": res.x,
                "ar_factors": ar_f,
                "ma_factors": ma_f,
                "loglik": ll,
                "loglik_pure": loglik_pure,
                "success": res.success,
                "R": data.R_list,
                "hess_inv": hess_inv,
                "slopes": slopes,
                # Provenance: which model the returned Sigma/loglik describe.
                "diag_load_lambda": float(diag_load_lambda or 0.0),
            }

    # Declared here (built lazily in the local branch below) so the optional
    # PSO-polish step after either branch can reuse the jax analytic objective.
    jax_fun = None
    # Same reason, plus the provenance block below reads it on every path.
    precond_scales = None

    if use_dual_annealing:
        n_ar   = d * p
        n_ma   = d * q
        n_chol = d * (d + 1) // 2

        da_bounds = []
        da_bounds.extend([(-20.0, 20.0)] * n_ar)
        da_bounds.extend([(-20.0, 20.0)] * n_ma)

        rows, cols = np.tril_indices(d)
        for i, j in zip(rows, cols):
            if i == j:
                da_bounds.append((-20.0, 20.0))
            else:
                da_bounds.append((-50.0, 50.0))

        da_bounds.extend([(-50.0, 50.0)] * d)

        x0 = (
            warm_theta
            if warm_theta is not None
            else prepare_initial_params_safe(data, p, q, rng=rng,
                                             ma_above_ar=ma_above_ar,
                                             zeta_mode=random_start_damping)
        )

        res = dual_annealing(
            neg_loglik,
            bounds=da_bounds,
            args=(data, p, q, slopes) + ridge_args,
            x0=x0,
            seed=int(rng.integers(1 << 31)),
            maxiter=n_restarts * 100,
            no_local_search=no_local_search,
            minimizer_kwargs={
                "method": method,
                "bounds": bounds,
                "options": {
                    "maxiter": 200,
                    "ftol": 1e-7,
                },
            },
        )

        _record(res, "dual_annealing")

        # Hybrid: pure SA explores cheaply (no_local_search), then a SINGLE
        # local BFGS polish refines the annealer's best point — instead of
        # dual_annealing's default per-iteration polish on every accepted step.
        if final_polish:
            polish_kwargs = {
                "method": method,
                "args": (data, p, q, slopes) + ridge_args,
            }
            if method == "L-BFGS-B":
                polish_kwargs["bounds"] = bounds
                polish_kwargs["options"] = {
                    "maxiter": 1000, "ftol": 1e-9, "maxfun": 5000,
                }
            else:  # BFGS
                polish_kwargs["options"] = {"maxiter": 1000, "gtol": 1e-9}

            res_polish = minimize(neg_loglik, res.x, **polish_kwargs)
            _record(res_polish, "final_polish")

    else:
        # Build restart seeds. Standard scheme (unchanged): seed 0 = warm exactly,
        # a few jittered-warm, the rest random. PLUS, whenever a valid warm start is
        # given for p>=2, ALWAYS append one underdamped low-frequency AR reset of it
        # (keeping the warm MA/cov/mu). A (2,0)-inherited warm sits near the zeta~1
        # critical-damping fold; without this extra seed the (2,1) fit cannot reach
        # the underdamped truth basin and collapses below (2,0). Appending (rather
        # than replacing) keeps it even at n_restarts=1 (the bootstrap path).
        seeds = []
        warm_ok = warm_theta is not None and len(warm_theta) == n_theta
        if local_only and warm_ok:
            # Pure local descent from the warm point. Every appended seed below
            # is a jump to another basin, and the winner across seeds is picked
            # on the MINIMIZED objective, so with the extra seeds in play a
            # "better" restart can score worse on the objective the caller
            # reports. Dropping them is what makes the finish monotone.
            n_restarts = 1
        for i in range(n_restarts):
            use_warm = (
                i == 0
                and warm_theta is not None
                and len(warm_theta) == n_theta
            )
            if use_warm:
                seeds.append(warm_theta.copy())
            elif warm_theta is not None and i < n_restarts // 2:
                seeds.append(warm_theta.copy()
                             + rng.normal(scale=1e-3, size=n_theta))
            else:
                seeds.append(prepare_initial_params_safe(
                    data, p, q, rng=rng, ma_above_ar=ma_above_ar,
                    zeta_mode=random_start_damping))
        if warm_ok and p >= 2 and not local_only:
            try:
                seeds.append(_underdamped_ar_reset(warm_theta, data, p, q, rng))
            except Exception:
                pass
            # meeting 7/17 #9 + 7/23 #2: also seed the OVERDAMPED basin (real
            # quasars are overdamped). Auto-on when the random start is asked to
            # span overdamped, so both the pure-random restarts AND the warm
            # reset seed cover the overdamped corner.
            if overdamped_reset or random_start_damping in ("over", "both"):
                try:
                    seeds.append(_overdamped_ar_reset(warm_theta, data, p, q, rng))
                except Exception:
                    pass

        # Optional JAX analytic gradient. scipy BFGS otherwise finite-differences
        # the gradient (~n_theta+1 Kalman passes per line-search step) and, with
        # gtol=1e-9, never converges so every fit burns maxiter. The JAX path
        # returns value+grad in one reverse-mode pass and runs with a looser
        # gtol. Only engaged when (p,q) and the prior config are representable
        # (jax_loglik.supports) AND a soft band prior is active (the legacy hard
        # hinge used when both band lambdas are None is not mirrored in JAX).
        jax_fun = None
        jax_prior = {}
        if use_jax_grad:
            from .jax_loglik import supports as _jax_supports, make_scipy_objective
            band_active = (ar_band_lambda is not None) or (ma_band_lambda is not None)
            if (_jax_supports(p, q, ar_pool_lambda, ma_pool_lambda, pole_zero_lambda)
                    and band_active):
                jax_prior = dict(
                    chol_ridge_lambda=chol_ridge_lambda,
                    chol_ridge_center=chol_ridge_center,
                    ar_band_lambda=ar_band_lambda, ma_band_lambda=ma_band_lambda,
                    chol_ridge_onesided=chol_ridge_onesided,
                    chol_ridge_center_hi=chol_ridge_center_hi,
                    ar_damping_lambda=ar_damping_lambda,
                    corr_ridge_lambda=corr_ridge_lambda,
                    diag_load_lambda=diag_load_lambda)
                try:
                    jax_fun = make_scipy_objective(
                        data, p, q, slopes=slopes, **jax_prior)
                except Exception as exc:
                    _log.warning(f"[fit] JAX grad unavailable ({exc}); using FD gradient")
                    jax_fun = None
            else:
                _log.warning("[fit] use_jax_grad requested but config unsupported "
                      "(need p in 1,2 / q in 0,1, soft band prior, no pooling); "
                      "using FD gradient")

        # Curvature preconditioning (opt-in via precondition="curvature").
        # See curvature_scales for why: the surface is stretched by about twelve
        # decades of curvature and BFGS starts from an identity inverse Hessian,
        # so without this most of the iteration budget is spent learning the
        # scaling instead of descending. Built ONCE, at the warm start when
        # there is one, and shared by every seed so the restarts stay comparable
        # to one another and to a run with the option off. Requires the JAX
        # objective, since the scales come from the exact Hessian diagonal of
        # the same penalized function the search minimizes.
        precond_scales = None
        if jax_fun is not None and precondition is not None:
            if isinstance(precondition, str):
                if precondition != "curvature":
                    raise ValueError(
                        f"precondition must be 'curvature', a scales array or "
                        f"None; got {precondition!r}")
                ref = np.asarray(warm_theta if warm_ok else seeds[0], float)
                try:
                    precond_scales = curvature_scales_at(
                        data, p, q, ref, slopes=slopes, **jax_prior)
                    _log.info(f"[fit] preconditioner: scales "
                          f"{precond_scales.min():.3e}.."
                          f"{precond_scales.max():.3e} "
                          f"(ref={'warm' if warm_ok else 'seed0'})")
                except Exception as exc:
                    # Never fatal: an unavailable preconditioner just means the
                    # old search, which is what every fit to date already ran.
                    _log.warning(f"[fit] preconditioner unavailable ({exc}); "
                          f"searching in raw coordinates")
                    precond_scales = None
            else:
                # Precomputed scales from curvature_scales_at. Length is checked
                # because a silently mismatched vector would broadcast into a
                # wrong-but-finite search rather than failing.
                cand = np.asarray(precondition, float)
                if cand.shape != (n_theta,):
                    raise ValueError(
                        f"precondition scales have shape {cand.shape}, "
                        f"expected ({n_theta},)")
                precond_scales = cand

        n_seeds = len(seeds)
        for i, theta0 in enumerate(seeds):

            if jax_fun is not None:
                # value+grad in one pass; looser gtol since the analytic grad is
                # exact (no FD noise to chase down to 1e-9).
                fun_i, x0_i, bounds_i = jax_fun, theta0, bounds
                if precond_scales is not None:
                    fun_i = precondition_objective(jax_fun, precond_scales)
                    x0_i = np.asarray(theta0, float) / precond_scales
                    if method == "L-BFGS-B" and bounds is not None:
                        bounds_i = [
                            (None if lo is None else lo / sc,
                             None if hi is None else hi / sc)
                            for (lo, hi), sc in zip(bounds, precond_scales)
                        ]
                res = minimize(
                    fun_i, x0_i, jac=True, method=method,
                    options=({"maxiter": maxiter, "ftol": 1e-9,
                              "maxfun": maxiter * 5, "gtol": jax_gtol}
                             if method == "L-BFGS-B"
                             else {"maxiter": maxiter, "gtol": jax_gtol}),
                    **({"bounds": bounds_i} if method == "L-BFGS-B" else {}),
                )
                if precond_scales is not None:
                    # Back to theta BEFORE _record: everything downstream of
                    # here (the tier gates, the reported theta, the hess_inv the
                    # SEs are read off) lives in theta space and must not know
                    # the search was reparameterized.
                    res = unscale_result(res, precond_scales)
                _record(res, f"Restart {i+1}/{n_seeds} [jax]")
                continue

            minimize_kwargs = {
                "method": method,
                "args": (data, p, q, slopes) + ridge_args,
            }

            if method == "L-BFGS-B":
                minimize_kwargs["bounds"] = bounds
                minimize_kwargs["options"] = {
                    "maxiter": maxiter,
                    "ftol": 1e-9,
                    "maxfun": maxiter * 5,
                }

            else:  # BFGS
                minimize_kwargs["options"] = {
                    "maxiter": maxiter,
                    "gtol": 1e-9,
                }

            res = minimize(
                neg_loglik,
                theta0,
                **minimize_kwargs,
            )

            _record(res, f"Restart {i+1}/{n_seeds}")

    # ------------------------------------------------------------------
    # PSO polish (meeting 7/17 #7). Optional honest global refinement: run a
    # bounded numpy particle swarm SEEDED at the best local optimum, then
    # sharpen the swarm's best with one local minimize. _record keeps it only
    # if it beats the incumbent, so this never lowers the fit. This is the
    # legitimate version of the deprecated overdamped-init arm -- it explores
    # basin structure derivative-free but, unlike that arm, does NOT drive
    # Sigma->0 (the 7/17 pso_diagnostic confirmed a real (2,1) loglik gap with
    # no Sigma collapse). Runs after the local restarts OR dual annealing.
    if pso_polish and best_result is not None:
        x_inc = np.asarray(best_result["theta"], float)

        def _pso_pen(x):
            try:
                v = neg_loglik(x, data, p, q, slopes, *ridge_args)
                return float(v) if np.isfinite(v) else 1e18
            except Exception:
                return 1e18

        # Per-block search half-width (log-param space): AR/MA wide, Cholesky
        # moderate, mu tight -- matches the standalone pso_diagnostic box.
        delta = np.concatenate([
            np.full(n_ar + n_ma, 5.0),
            np.full(n_chol, 2.0),
            np.full(d, 0.5),
        ])
        x_pso = None
        try:
            x_pso, _ = _pso_search(
                _pso_pen, x_inc, delta,
                n_particles=pso_particles, iters=pso_iters,
                seed=int(rng.integers(1 << 31)),
                w=pso_w, c1=pso_c1, c2=pso_c2, topology=pso_topology)
        except Exception as exc:
            _log.warning(f"[fit] PSO polish failed ({exc}); keeping incumbent")

        if x_pso is not None:
            # Sharpen the swarm best with one local minimize (jax grad when the
            # config supports it, FD otherwise). Same optimiser/objective as the
            # production restarts, so this opens no Sigma-collapse basin the
            # warm-start path would not already have reached.
            if jax_fun is not None:
                res_pso = minimize(
                    jax_fun, x_pso, jac=True, method=method,
                    options=({"maxiter": maxiter, "ftol": 1e-9,
                              "maxfun": maxiter * 5, "gtol": jax_gtol}
                             if method == "L-BFGS-B"
                             else {"maxiter": maxiter, "gtol": jax_gtol}),
                    **({"bounds": bounds} if method == "L-BFGS-B" else {}))
            else:
                res_pso = minimize(
                    neg_loglik, x_pso,
                    args=(data, p, q, slopes) + ridge_args, method=method,
                    options=({"maxiter": maxiter, "ftol": 1e-9,
                              "maxfun": maxiter * 5}
                             if method == "L-BFGS-B"
                             else {"maxiter": maxiter, "gtol": 1e-9}),
                    **({"bounds": bounds} if method == "L-BFGS-B" else {}))
            _record(res_pso, "pso_polish")

    if best_result is None:
        raise RuntimeError("All optimisation restarts failed.")

    # Per-call restart provenance. `winning_restart` is the label of the restart
    # whose optimum is being returned, so a fit that won on its first try and
    # one that needed the last of four are distinguishable after the fact.
    best_result["restarts"] = {
        "budget": int(n_restarts),
        "attempted": int(n_attempted),
        "finite": int(n_finite),
        "winning_restart": best_label,
        "winning_nit": int(best_nit),
        "winning_nfev": int(best_nfev),
        "success": bool(best_result.get("success", False)),
        # Which function decided the winner, so a fit ranked on the reported
        # likelihood is distinguishable after the fact from one ranked on the
        # objective it happened to minimize.
        # "objective_fallback" means restart_score could not score a single
        # restart and the objective-ranked point is coming back as a last
        # resort, so the ranking fix did not apply to this fit at all.
        "ranked_by": ("objective" if restart_score is None else
                      ("restart_score" if best_tier >= 1
                       else "objective_fallback")),
        "unscorable": int(n_unscorable),
        # Rank band of the winner (see _quality_tier): 3 interior+stationary,
        # 2 interior+unfinished, 1 ill conditioned or gates off, 0 unscorable.
        # A reported fit at tier 1 with the gates ON is on the Sigma->0
        # boundary and every restart was, which is the case where the MLE does
        # not exist rather than the case where the search failed.
        "winning_tier": int(best_tier),
        "tier_gated": bool(restart_cond_max is not None or restart_grad is not None),
        # Whether the search ran in curvature-scaled coordinates. A fit with
        # this False ran the historical un-preconditioned search, so the two
        # populations stay separable in any later comparison of the arms.
        "preconditioned": bool(precond_scales is not None),
    }

    return best_result
