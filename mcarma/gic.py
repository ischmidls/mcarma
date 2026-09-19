"""
mcarma/gic.py
=============
Generalized Information Criterion (Konishi & Kitagawa) for the penalized
mCARMA estimator, so that model selection uses a criterion that matches the
penalized estimator instead of assuming plain maximum likelihood.

Why AICc is not enough
----------------------
AICc scores ``-2 loglik + 2 k n/(n-k-1)`` and its ``2k`` bias term is derived
under maximum likelihood with the model correctly specified. Our estimator is
penalized, and the fits can be mildly misspecified, so the
effective number of parameters is not ``k``. Konishi and Kitagawa's GIC replaces
``k`` with a bias correction computed from the estimator's own influence
functions and works for any estimator defined by an estimating equation,
including a penalized one.

The criterion
-------------
Let the estimator maximize ``L(theta) = sum_i log f_i(theta) - p_lambda(theta)``
over the ``n`` scalar Kalman innovations (band-epochs). Write

  * ``s_i = d log f_i / d theta``            per-observation model score,
  * ``S``  = the ``(n, k)`` matrix of the ``s_i`` (the score seam exposed by
             ``build_objective(..., with_scores=True)``),
  * ``K = S^T S = sum_i s_i s_i^T``          the outer-product 'meat' matrix,
  * ``J``  = the penalized-objective Hessian ``d^2 (-L) / d theta^2``
             (the analytic ``hess_j`` from ``build_objective``), the 'bread',
  * ``g_lambda = sum_i s_i``                 which equals the penalty gradient
             at the optimum by the penalized score equation, read directly off
             the column sums of ``S``.

Then the GIC effective dimension is

  ``df = tr(J^{-1} K) - (1/n) * g_lambda^T J^{-1} g_lambda``

and ``GIC = -2 * loglik + 2 * df`` with ``loglik`` the pure (unpenalized)
log-likelihood ``sum_i log f_i(theta_hat)``. The second df term is ``O(1/n)`` and
vanishes wherever the penalty is inactive (``g_lambda = 0``); it only bites in
the near-singular (2,1) tail where the penalty binds. In the correct-model
maximum-likelihood limit ``g_lambda -> 0`` and ``K -> J -> Fisher``, so
``df -> k`` and ``GIC -> AIC``. That limit is the validation test
(``tests/test_gic.py::test_ml_limit_algebra``).

This module is refit-free: it consumes a converged ``theta`` and the same data
the fit saw, and does no optimization.
"""
from __future__ import annotations

import numpy as np


def _bread_meat(data, p, q, theta, slopes=None, prior=None):
    """Return (loglik, J, K, g_lambda, n_obs) at ``theta``.

    ``loglik`` is the pure unpenalized log-likelihood, ``J`` the penalized-
    objective Hessian, ``K = S^T S`` the score outer product, ``g_lambda`` the
    column sum of the score matrix (the penalty gradient at the optimum), and
    ``n_obs`` the number of scalar innovations.
    """
    import jax
    import jax.numpy as jnp
    from mcarma.jax_loglik import build_objective

    prior = dict(prior or {})
    value_fn, _vg, hess_fn, ll_vec_fn = build_objective(
        data, p, q, slopes=slopes, with_scores=True, **prior)

    x = jnp.asarray(np.asarray(theta, dtype=float))
    ll_vec = np.asarray(ll_vec_fn(x), dtype=float)          # (n_obs,)
    loglik = float(ll_vec.sum())
    n_obs = int(ll_vec.shape[0])

    # Per-observation score matrix S (n_obs, k). n_theta << n_obs, so forward
    # mode (jacfwd) is far cheaper than reverse over every innovation.
    S = np.asarray(jax.jacfwd(ll_vec_fn)(x), dtype=float)   # (n_obs, k)
    K = S.T @ S                                             # (k, k) meat
    g_lambda = S.sum(axis=0)                                # (k,) penalty grad

    J = np.asarray(hess_fn(x), dtype=float)                 # (k, k) bread
    J = 0.5 * (J + J.T)
    return loglik, J, K, g_lambda, n_obs


def _solve_spd(J, B):
    """Solve ``J X = B`` with J symmetric, ridging onto PD if needed.

    Returns ``(X, ridged)`` where ``ridged`` flags whether a Cholesky ridge was
    applied because J was not numerically positive definite.
    """
    J = 0.5 * (J + J.T)
    try:
        L = np.linalg.cholesky(J)
        X = np.linalg.solve(J, B)
        return X, False
    except np.linalg.LinAlgError:
        # Near-singular / indefinite bread: ridge by a small multiple of the
        # mean positive curvature so the trace stays finite and reported.
        w = np.linalg.eigvalsh(J)
        scale = np.median(np.abs(w[w > 0])) if np.any(w > 0) else 1.0
        eps = 1e-6 * max(scale, 1.0)
        Jr = J + eps * np.eye(J.shape[0])
        X = np.linalg.solve(Jr, B)
        return X, True


def compute_gic(data, p, q, theta, slopes=None, prior=None):
    """GIC and its effective dimension for one penalized mCARMA fit.

    Parameters
    ----------
    data : the fit's data object (same one passed to the fitter / jax_loglik).
    p, q : CARMA order.
    theta : converged parameter vector (length n_ar + n_ma + n_chol + d).
    slopes : optional per-band linear slopes, as passed to ``build_objective``.
    prior : dict of the penalty lambdas / centers used at fit time (the same
        keyword arguments given to ``build_objective``); ``None`` = no penalty.

    Returns
    -------
    dict with the pure ``loglik``; nominal parameter count ``k``; the GIC
    effective dimension ``df_gic`` (full, with the penalty term) and
    ``df_leading`` (the ``tr(J^{-1}K)`` piece alone); ``gic``, ``aic``, and the
    small-sample ``aicc`` for reference; and diagnostics ``penalty_term``,
    ``g_norm`` (size of the penalty gradient, ~0 where the penalty is inactive),
    ``bread_ridged`` (whether J needed a PD ridge), and ``n_obs``.
    """
    loglik, J, K, g_lambda, n_obs = _bread_meat(
        data, p, q, theta, slopes=slopes, prior=prior)
    k = int(J.shape[0])

    JinvK, ridged = _solve_spd(J, K)
    df_leading = float(np.trace(JinvK))

    Jinv_g, _ = _solve_spd(J, g_lambda)
    penalty_term = float(g_lambda @ Jinv_g) / n_obs        # (1/n) g^T J^-1 g
    df_gic = df_leading - penalty_term

    aic = -2.0 * loglik + 2.0 * k
    aicc = (aic + (2.0 * k * (k + 1)) / (n_obs - k - 1)
            if n_obs - k - 1 > 0 else float("nan"))
    gic = -2.0 * loglik + 2.0 * df_gic

    return dict(
        loglik=loglik, k=k, n_obs=n_obs,
        df_gic=df_gic, df_leading=df_leading, penalty_term=penalty_term,
        gic=gic, aic=aic, aicc=aicc,
        g_norm=float(np.linalg.norm(g_lambda)),
        bread_ridged=bool(ridged),
    )
