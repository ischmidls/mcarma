"""fit -> score -> standard-errors, composed into one shared call.

Every driver (real-data process_quasars, the bigindep sim, the LSST downsample) runs
the same core per-order recipe: fit an MCARMA(p,q), read the PURE maximized
log-likelihood (not the penalized objective), score AICc on it, and attach a
standard error. Historically each open-coded that recipe, which is how the AICc got
scored on the penalized objective in one place and the pure one in another, and how the
SE backend drifted between drivers. ``fit_and_score`` is the single composition.

The SE backend is selected by ``se``:

  * ``"hess_inv_diag"`` -- se_theta = sqrt(max(diag(BFGS hess_inv), 0)), and None when
    the fit did not expose a hess_inv (no FD fallback). This is the historical LSST
    driver behavior, kept byte-identical so the existing corpus stays comparable.
  * any ``standard_errors`` source (``"auto"``/``"polish"``/``"exact"``/``"hess_inv"``)
    -- routes through mcarma.inference.standard_errors (the trustworthy analytic
    repolish + JAX-HVP path under "auto"), returning the full covariance too.
"""
import numpy as np

from mcarma.fit import fit
from mcarma.model_utils import compute_aicc, count_params
from mcarma.inference import standard_errors


def fit_and_score(data, p, q, *, warm_theta=None, prior_kwargs=None,
                  fit_kwargs=None, se="hess_inv_diag", use_jax=False,
                  hess_method="rich", n_obs=None):
    """Fit MCARMA(p,q), score AICc on the pure loglik, and attach SEs.

    Parameters mirror what the drivers already pass: ``warm_theta`` and
    ``prior_kwargs`` go to ``fit`` alongside ``fit_kwargs`` (n_restarts, seed,
    maxiter, random_start_damping, pso_polish, ...) and ``use_jax`` (-> fit's
    ``use_jax_grad``). ``n_obs`` defaults to ``data.n``.

    Returns a dict: ``result`` (raw fit dict), ``theta``, ``ll`` (PURE), ``ll_pen``,
    ``k``, ``aicc``, ``success``, ``hess_inv``, ``se_theta``, ``cov``, ``pd``,
    ``se_backend``, ``grad_norm_map``, ``theta_se``. ``cov``/``pd``/``theta_se`` are
    None under ``se="hess_inv_diag"`` (that mode reports only the diagonal SE)."""
    d = data.d
    if n_obs is None:
        n_obs = data.n
    k = count_params(d, p, q)
    result = fit(data, p, q, warm_theta=warm_theta, use_jax_grad=use_jax,
                 **(fit_kwargs or {}), **(prior_kwargs or {}))

    # AICc on the PURE maximized log-likelihood, not the penalized objective:
    # result["loglik"] is pure_ll - penalty - ridge, so scoring it would leak the
    # (always-nonzero) prior penalty into every AICc. loglik_pure removes that.
    ll_pen = result["loglik"]
    ll = result.get("loglik_pure", ll_pen)
    aicc = compute_aicc(ll, k, n_obs)

    theta = np.asarray(result["theta"], dtype=float)
    hess_inv = result.get("hess_inv")
    out = {
        "result": result, "theta": theta, "ll": float(ll), "ll_pen": float(ll_pen),
        "k": int(k), "aicc": float(aicc), "success": bool(result.get("success", False)),
        "hess_inv": hess_inv, "se_theta": None, "cov": None, "pd": False,
        "se_backend": None, "grad_norm_map": None, "theta_se": theta,
    }

    if se == "hess_inv_diag":
        # Raw BFGS inverse-Hessian diagonal (the legacy LSST SE). None when the fit
        # exposed no hess_inv -- NO FD fallback, so behavior is byte-identical.
        if hess_inv is not None:
            se_theta = np.sqrt(np.maximum(np.diag(np.asarray(hess_inv)), 0)).tolist()
            out["se_theta"] = se_theta
            out["se_backend"] = "bfgs_hess_inv_diag"
        return out

    # Otherwise route through the shared SE dispatcher (full covariance).
    seres = standard_errors(theta, data, p, q, result=result,
                            prior_kwargs=prior_kwargs, source=se, use_jax=use_jax,
                            hess_method=hess_method)
    cov = seres["cov"]
    out["theta_se"] = np.asarray(seres["theta_se"])
    out["cov"] = cov
    out["pd"] = bool(seres["pd"])
    out["se_backend"] = seres["se_backend"]
    out["grad_norm_map"] = seres["grad_norm_map"]
    if cov is not None:
        out["se_theta"] = np.sqrt(np.maximum(np.diag(np.asarray(cov)), 0)).tolist()
    return out
