"""Canonical standard-error machinery shared by every driver.

Standard errors here come from the curvature of the PENALIZED log-likelihood at
its maximum. The penalties are regularizers, not a prior, and the inverse
Hessian is a sandwich-free frequentist covariance, not a posterior. Names in
this module that read "MAP" are historical; they mark the penalized optimum.

Historically the sim study (fits/21sim/sim_study.py) carried its own copies of the
SE code -- FD Hessian, the re-polish, the analytic JAX-HVP path -- while the real-data
Stage-5 estimator (fits/process_quasars/penalized_hessian.py) and the LSST driver
(fits/21sim/fit_lsst_downsample.py) each reported SEs a slightly different way. The most
common mistake this project makes is reporting the WRONG SE: the BFGS ``hess_inv`` is a
descent preconditioner at the optimizer's stopping point, not the curvature at a
stationary optimum, so it under-covers. This module is the single home for the SE estimators
and a ``standard_errors`` dispatcher so no driver silently falls back to the raw
``hess_inv``.

The estimator bodies (``approx_hessian``, ``mle_cov``, ``polished_cov``, ``_jax_se_objs``,
the ``extract_*`` reporters) are relocated verbatim from sim_study.py, so results are
byte-identical to the pre-refactor sim study. ``standard_errors`` reproduces sim_study's
``--se-source`` branch exactly, and adds ``source="auto"`` = the analytic repolish +
JAX-HVP Hessian path (the trustworthy SE), so callers can opt every path onto it.

The real-data path (penalized_hessian.py) already routes its Hessian through the same
``jax_loglik.make_scipy_hessian_hvp`` backend but keeps an FD repolish by design; it is
left untouched here to preserve the published real-data SEs. Converging it onto this
module's polish is a deliberate, separately-validated change, not a side effect."""
import warnings

import numpy as np

from mcarma.fit import neg_loglik
from mcarma.reporting import chol_jacobian, chol_theta_to_sigma


# ===========================================================================
# Hessian primitives
# ===========================================================================
def approx_hessian(func, x0, args=(), epsilon=None):
    """Single-step central finite-difference Hessian of a scalar ``func``."""
    x0 = np.asarray(x0, dtype=float)
    n = x0.size
    if epsilon is None:
        epsilon = np.power(np.finfo(float).eps, 1.0 / 3.0)
    H = np.zeros((n, n))
    f0 = func(x0, *args)
    for i in range(n):
        hi = epsilon * max(abs(x0[i]), 1.0)
        xip = x0.copy(); xip[i] += hi
        xim = x0.copy(); xim[i] -= hi
        H[i, i] = (func(xip, *args) - 2.0 * f0 + func(xim, *args)) / hi ** 2
        for j in range(i + 1, n):
            hj = epsilon * max(abs(x0[j]), 1.0)
            xpp = x0.copy(); xpp[i] += hi; xpp[j] += hj
            xpm = x0.copy(); xpm[i] += hi; xpm[j] -= hj
            xmp = x0.copy(); xmp[i] -= hi; xmp[j] += hj
            xmm = x0.copy(); xmm[i] -= hi; xmm[j] -= hj
            H[i, j] = (func(xpp, *args) - func(xpm, *args)
                       - func(xmp, *args) + func(xmm, *args)) / (4.0 * hi * hj)
            H[j, i] = H[i, j]
    return 0.5 * (H + H.T)


def mle_cov(theta_fit, data, p, q, prior_kwargs=None, use_jax=False):
    """Invert the numerical Hessian of the *penalized* neg_loglik at theta_fit.

    The Hessian must be of the SAME objective the fit optimized, so the prior
    kwargs are forwarded (previously they were dropped, giving a prior-blind
    Hessian inconsistent with the penalized fit). pinv fallback when non-PD.

    ``use_jax`` is accepted for call-site symmetry but the Hessian stays FD: the
    jax jacfwd-jacrev Hessian OOMs at study scale (see polished_cov)."""
    pk = prior_kwargs or {}
    pen = lambda th: neg_loglik(th, data, p, q, **pk)
    H = approx_hessian(pen, theta_fit)
    evals = np.linalg.eigvalsh(H)
    if np.any(evals <= 0):
        return np.linalg.pinv(H), False
    try:
        return np.linalg.inv(H), True
    except np.linalg.LinAlgError:
        return np.linalg.pinv(H), False


def _fd_grad_norm(fn, x, eps=1e-5):
    """L2 norm of a central finite-difference gradient (polish stop test)."""
    g = np.empty_like(x, dtype=float)
    for i in range(x.size):
        xp = x.copy(); xp[i] += eps
        xm = x.copy(); xm[i] -= eps
        g[i] = (fn(xp) - fn(xm)) / (2.0 * eps)
    return float(np.linalg.norm(g))


def _jax_se_objs(data, p, q, pk):
    """(value_and_grad, hessian) JAX callables for the SE path, or (None, None).

    Only when jax_loglik can represent the (p,q)+prior config (p in 1,2; q in
    0,1; no pooling/pole-zero) AND a soft band prior is active (the legacy hinge
    is not mirrored). Used by polished_cov / mle_cov to replace the FD re-polish
    + FD Hessian -- exact, ~80x faster, identical SEs (validated _diag_jaxhess)."""
    try:
        from mcarma import jax_loglik
    except Exception:
        return None, None
    if not jax_loglik.supports(p, q, pk.get("ar_pool_lambda"),
                               pk.get("ma_pool_lambda"), pk.get("pole_zero_lambda")):
        return None, None
    if pk.get("ar_band_lambda") is None and pk.get("ma_band_lambda") is None:
        return None, None
    keys = ("chol_ridge_lambda", "chol_ridge_center", "ar_band_lambda",
            "ma_band_lambda", "chol_ridge_onesided", "chol_ridge_center_hi",
            "ar_damping_lambda",
            # Mirrored in jax_loglik.build_objective. Dropping them would return
            # the value/grad/Hessian of a DIFFERENT objective than the fit used
            # (unridged, unloaded) with nothing raised to say so.
            "corr_ridge_lambda", "diag_load_lambda")
    pj = {k: pk[k] for k in keys if k in pk}
    try:
        # Memory-safe HVP-loop analytic Hessian (not the vectorized jacfwd-jacrev,
        # which OOMs materializing all n_theta tangents through the Kalman scan).
        return (jax_loglik.make_scipy_objective(data, p, q, **pj),
                jax_loglik.make_scipy_hessian_hvp(data, p, q, **pj))
    except Exception:
        return None, None


def polished_cov(theta_fit, data, p, q, prior_kwargs=None,
                 gtol=1e-2, max_rounds=8, hess_method="rich", use_jax=False):
    """Trustworthy SE source for the recovery table (the §3.2 coverage fix).

    The fit's BFGS ``hess_inv`` is a descent preconditioner evaluated at the
    optimizer's stopping point, NOT the curvature at a stationary optimum -- it
    makes the SEs too small (under-coverage). This mirrors the real-data
    ``penalized_hessian`` estimator instead:

      1. polish ``theta_fit`` to the optimum of the SAME penalized objective the
         fit used (matched ``prior_kwargs``), alternating L-BFGS-B / BFGS until
         the gradient is small -- the quadratic expansion is only valid at
         grad~0;
      2. recompute a Hessian AT that optimum and invert it (pinv if non-PD).

    With ``use_jax`` the re-polish runs on the JAX analytic value+grad (jac=True)
    and the Hessian is the JAX autodiff Hessian -- exact and ~80x faster than the
    FD re-polish + FD Hessian, with statistically identical SEs (validated). This
    is a SPEED fix only; the coverage itself is addressed by the bootstrap. Falls
    back to FD whenever the config is unsupported.

    Returns ``(theta_map, cov, pd, grad_norm, backend)``; the polished
    ``theta_map`` is also the proper point estimate for recovery (the penalized
    optimum, not the loose stop).

    What ``pd`` does and does not mean
    ----------------------------------
    ``pd`` tests the FULL Hessian for positive definiteness. That is the right
    test only at an interior optimum. When a coordinate settles on the Sigma
    soft-box floor or a resolvable-band edge, the optimum is constrained, and a
    direction the penalty wall blocks can carry negative curvature while the
    point is still a maximum over the feasible set. Measured on the 134-object
    Stripe 82 fits, the full Hessian is PD at 13 of the 89 converged objects
    while the Hessian restricted to the unconstrained coordinates is PD at all
    89. So ``pd=False`` is not by itself evidence of a bad fit; check whether
    coordinates are on a wall before reading it that way. ``cov`` uses ``pinv``
    when ``pd`` is False, so the returned SEs are still usable, and the SEs of
    on-wall coordinates are the ones to distrust."""
    import scipy.optimize as so
    pk = prior_kwargs or {}
    pen = lambda th: float(neg_loglik(th, data, p, q, **pk))

    jax_vg, jax_hess = (_jax_se_objs(data, p, q, pk) if use_jax else (None, None))

    if jax_vg is not None:
        # JAX path: re-polish with the analytic value+grad (jac=True), and take
        # the Hessian from the memory-safe HVP-loop analytic Hessian (jax_hess) --
        # exact curvature of the penalized objective in O(n_theta) forward-over-
        # reverse passes, replacing the O(n_theta^2) single-step FD approx_hessian
        # (~660s -> seconds, no step-size error). Falls back to FD only if the
        # HVP Hessian could not be built.
        fun = jax_vg                      # returns (value, grad) -> jac=True
        grad_norm = lambda x: float(np.linalg.norm(jax_vg(x)[1]))
        if jax_hess is not None:
            hess_at = jax_hess
            backend = "jax_repolish+jax_hvp_hess"
        else:
            hess_at = lambda x: approx_hessian(pen, x)
            backend = "jax_repolish+fd_approx_hess"
    else:
        # FD fallback. Single-step central-FD Hessian (approx_hessian) by
        # default; numdifftools Richardson with hess_method="rich".
        fun = None
        grad_norm = lambda x: _fd_grad_norm(pen, x)
        if hess_method == "rich":
            try:
                import numdifftools as nd
                hess_at = lambda x: np.asarray(nd.Hessian(pen, method="central",
                                                          step=None)(x))
                backend = "fd_repolish+fd_richardson_hess"
            except Exception:
                hess_at = lambda x: approx_hessian(pen, x)
                backend = "fd_repolish+fd_approx_hess"
        else:
            hess_at = lambda x: approx_hessian(pen, x)
            backend = "fd_repolish+fd_approx_hess"

    x = np.array(theta_fit, dtype=float)
    f_prev = pen(x)
    gn = grad_norm(x)
    for r in range(max_rounds):
        meth = "BFGS" if (r % 2) else "L-BFGS-B"
        opt = (dict(maxiter=400) if meth == "BFGS"
               else dict(maxiter=400, ftol=1e-15, gtol=1e-12, eps=1e-6, maxls=50))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if fun is not None:
                res = so.minimize(fun, x, jac=True, method=meth, options=opt)
            else:
                res = so.minimize(pen, x, method=meth, options=opt)
        gn = grad_norm(res.x)
        improved = f_prev - res.fun
        x = res.x
        f_prev = res.fun
        if gn < gtol or (improved < 1e-7 and r >= 1):
            break

    H = hess_at(x)
    H = 0.5 * (H + H.T)
    evals = np.linalg.eigvalsh(H)
    pd = bool(np.all(evals > 0))
    cov = np.linalg.inv(H) if pd else np.linalg.pinv(H)
    return x, cov, pd, gn, backend


# ===========================================================================
# Reported-parameter extraction  (alpha_{1,1}, b_{1,1}, rho_{12} with SEs)
# ===========================================================================
def extract_alpha11(theta, cov):
    """alpha_{1,1} = first AR polynomial coeff of band 1 = exp(theta[0]).

    theta stores log a_{i,j} (Jones form); for a single AR factor of order <=2
    the first polynomial coeff equals the first factor coeff. Delta-method SE
    se(alpha) = alpha * se(log a). Log-scale value/SE returned for reference."""
    log_a = float(theta[0])
    se_log = float(np.sqrt(max(cov[0, 0], 0.0)))
    raw = float(np.exp(log_a))
    se_raw = float(raw * se_log)   # delta method: d(exp x)/dx = exp x
    return {"log": log_a, "log_se": se_log, "raw": raw, "raw_se": se_raw}


def extract_b11(theta, cov, p, q, d):
    """b_{1,1} = first MA factor coeff of band 1 (q>=1 only). Band-1 MA log-coeff
    is theta[p*d]; raw = exp(theta[p*d]); delta-method SE = raw * se(log b).
    Returns nan fields when q < 1."""
    if q < 1:
        return {"log": np.nan, "log_se": np.nan, "raw": np.nan, "raw_se": np.nan}
    idx = p * d
    log_b = float(theta[idx])
    se_log = float(np.sqrt(max(cov[idx, idx], 0.0)))
    raw = float(np.exp(log_b))
    se_raw = float(raw * se_log)
    return {"log": log_b, "log_se": se_log, "raw": raw, "raw_se": se_raw}


def extract_rho12(theta, cov, p, q, d, diag_load_lambda=0.0):
    """rho_{12} = corr(Sigma)[0,1] with delta-method SE propagated through the
    Cholesky block of theta. Returns {"value", "se"}.

    diag_load_lambda : float, default 0.0
        When diagonal loading is active the likelihood is evaluated at the LOADED
        covariance (fit.py neg_loglik / jax_loglik._state_space),

            Sigma_eff = Sigma + lam * (tr Sigma / d) * I,

        so the correlation implied by the raw theta Cholesky block is NOT the one
        the fit optimized. Reading the raw scale reverses the sign of the
        coherence-bias conclusion, because the estimator raises the unloaded
        correlation to offset the identity it knows will be added. Pass the
        active lambda and this returns the loaded-scale correlation. Default 0.0
        reproduces the unloaded value and SE exactly.
    """
    chol_start = p * d + q * d
    n_chol = d * (d + 1) // 2
    theta_chol = theta[chol_start:chol_start + n_chol]
    cov_chol = cov[chol_start:chol_start + n_chol,
                   chol_start:chol_start + n_chol]

    J = chol_jacobian(theta_chol, d)
    cov_sig = J @ cov_chol @ J.T            # cov of vech(Sigma), tril order
    _, Sigma = chol_theta_to_sigma(theta_chol, d)

    # tril(d) order: index of (i,j) is i*(i+1)//2 + j, so diag (i,i) -> i*(i+3)//2.
    lam = float(diag_load_lambda or 0.0)
    a, c, b = Sigma[0, 0], Sigma[1, 0], Sigma[1, 1]
    # Loading shifts BOTH variances by the same s = lam*tr(Sigma)/d and leaves the
    # covariance c untouched, so it can only shrink |rho| toward zero.
    s = lam * float(np.trace(Sigma)) / d if lam > 0.0 else 0.0
    A, B = a + s, b + s
    denom = np.sqrt(max(A * B, 1e-300))
    rho = c / denom

    # d(rho)/d(vech Sigma). s depends on EVERY diagonal entry, so with loading on
    # the gradient reaches past the (0,0),(1,0),(1,1) corner: dA/dSigma_ii =
    # lam/d + [i==0], dB/dSigma_ii = lam/d + [i==1].
    g = np.zeros(n_chol)
    g[1] = 1.0 / denom                                    # d/dc
    dA, dB = -rho / (2.0 * A), -rho / (2.0 * B)
    for i in range(d):
        g[i * (i + 3) // 2] = dA * (lam / d + (1.0 if i == 0 else 0.0)) \
                            + dB * (lam / d + (1.0 if i == 1 else 0.0))
    var = float(g @ cov_sig @ g)
    return {"value": float(rho), "se": float(np.sqrt(max(var, 0.0)))}


# ===========================================================================
# The one shared SE call
# ===========================================================================
def standard_errors(theta, data, p, q, *, result=None, prior_kwargs=None,
                    source="auto", use_jax=False, hess_method="rich"):
    """Single entry point for the parameter covariance / SEs of a fit.

    Reproduces sim_study's ``--se-source`` branch exactly and adds ``"auto"``:

      * ``"auto"``  -> repolish + (JAX-HVP if ``use_jax`` and supported, else FD)
                       Hessian. The trustworthy SE; nothing silently uses hess_inv.
      * ``"polish"`` -> same as "auto" (kept for the explicit --se-source value).
      * ``"hess_inv"`` -> the fit's BFGS inverse Hessian (``result["hess_inv"]``),
                       symmetrized; falls through to "exact" if it is unavailable.
      * ``"exact"`` -> FD Hessian of the penalized objective at ``theta`` (mle_cov).

    Returns ``dict(theta_se, cov, pd, se_backend, grad_norm_map)``. ``theta_se`` is
    the polished optimum for the polish/auto paths (the proper point estimate)
    and the input ``theta`` otherwise. ``grad_norm_map`` is None off the polish
    path; the ``_map`` in that key name is historical."""
    pk = prior_kwargs
    theta_se = np.asarray(theta)
    cov, pd, gn, se_backend = None, False, None, None
    src = "polish" if source == "auto" else source
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if src == "polish":
            theta_se, cov, pd, gn, se_backend = polished_cov(
                theta, data, p, q, pk, use_jax=use_jax, hess_method=hess_method)
        elif src == "hess_inv":
            hi = (result or {}).get("hess_inv")
            if hi is not None:
                cov = np.asarray(hi, dtype=float)
                cov = 0.5 * (cov + cov.T)
                pd = bool(np.all(np.linalg.eigvalsh(cov) > 0))
                se_backend = "bfgs_hess_inv"
        if cov is None:    # "exact", or hess_inv unavailable (e.g. L-BFGS-B)
            cov, pd = mle_cov(theta_se, data, p, q, pk, use_jax=use_jax)
            se_backend = "fd_approx_mle_cov"
    return dict(theta_se=np.asarray(theta_se), cov=cov, pd=bool(pd),
                se_backend=se_backend, grad_norm_map=gn)
