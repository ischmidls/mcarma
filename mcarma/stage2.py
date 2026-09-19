# -*- coding: utf-8 -*-
"""Stage-2 MLE finish: an ordinary-likelihood optimization, started from the
stage-1 penalized point, that is not allowed to end below where it started.

Why this module exists
----------------------
Stage 1 optimizes a penalized likelihood of the *loaded* model (the driving
covariance enters the state space as Sigma + lam * mean(diag Sigma) * I).
Stage 2 optimizes the ordinary likelihood of the *unloaded* model. Keeping
whichever of the two points scores higher on the ordinary likelihood makes the
reported number monotone, but it does not make the reported point an MLE: the
stage-1 point was never optimized against the ordinary likelihood and need not
be stationary for it.

So retry stage 2 instead of falling back to stage 1. Two things let the old
single attempt finish below its own starting value, and both are settings
rather than facts about the likelihood surface:

1. The search minimized the JAX objective while the result was scored with the
   Kalman ``neg_loglik``. The two agree to six decimals at a sane theta and not
   at all in the extremes, so descent in one is not descent in the other.
2. ``fit`` appends damping-reset seeds for p >= 2 and returned the best seed by
   the minimized objective, so the returned point could be another basin's
   optimum that scores worse than the start on the reported likelihood. The
   multi-seed rung now passes ``restart_score=plain_ll``, so the seeds are
   ranked on the same function the acceptance test uses and a better seed is no
   longer discarded unscored. The escalation below still covers what ranking
   cannot: a rung whose every seed lands badly.

The ladder below removes them one at a time and ends with a search that
minimizes exactly the function the result is scored on, from exactly the
starting point, with nothing else in play.

The caller gets ``converged`` and ``attempt`` so a residual failure stays
visible: a point no attempt could improve is returned as the starting point
with ``converged=False``, not silently relabeled an MLE.
"""
import logging

import numpy as np
from scipy.optimize import minimize

from .fit import fit, neg_loglik, curvature_scales_at

_log = logging.getLogger(__name__)

# Explicit zeros, not omitted: with both band lambdas None, neg_loglik takes the
# legacy near-hard hinge instead of the soft band prior, so dropping the kwargs
# swaps one penalty for another rather than removing it.
PLAIN_PK = {"ar_band_lambda": 0.0, "ma_band_lambda": 0.0}

# The ordered attempt names. Least restrictive first, so an arm that never had
# a problem reproduces what production already ran and stays comparable.
LADDER = ("jax_multi", "jax_local", "kalman_local", "nelder_mead")


def plain_ll(theta, data, p, q, slopes=None):
    """The ordinary log-likelihood: no penalty, no hinge, no diagonal loading.

    This is the one scoring function. Every number the ladder compares comes
    from here, so no comparison is ever made across two different functions.

    ``slopes`` is part of the objective, not a preprocessing detail: under
    --slopes-in-likelihood the per-band trend is estimated jointly rather than
    subtracted first, so a score taken without it is a DIFFERENT function. It
    defaults to None because the simulation study has no trends, but the real
    Stripe 82 path does, and dropping it there would silently reintroduce the
    cross-function comparison this module exists to eliminate.
    """
    try:
        v = float(-neg_loglik(np.asarray(theta, float), data, p, q,
                              slopes=slopes, **PLAIN_PK))
    except Exception:
        return float("nan")
    return v if (np.isfinite(v) and v > -1e9) else float("nan")


def plain_grad_norm(theta, data, p, q, use_jax=True, slopes=None):
    """Sup-norm of the ordinary-likelihood gradient at ``theta``.

    Evidence for the stationarity claim, reported rather than enforced, so it
    returns NaN instead of raising when no gradient is available.
    """
    theta = np.asarray(theta, float)
    if use_jax:
        try:
            from .jax_loglik import supports, make_scipy_objective
            if supports(p, q):
                fun = make_scipy_objective(data, p, q, slopes=slopes,
                                           **PLAIN_PK)
                _, g = fun(theta)
                g = np.asarray(g, float)
                if np.isfinite(g).all():
                    return float(np.max(np.abs(g)))
        except Exception:
            pass
    try:
        f0 = float(neg_loglik(theta, data, p, q, slopes=slopes, **PLAIN_PK))
        if not np.isfinite(f0):
            return float("nan")
        step = 1e-5 * np.maximum(1.0, np.abs(theta))
        g = np.empty_like(theta)
        for i in range(theta.size):
            tp = theta.copy()
            tp[i] += step[i]
            g[i] = (float(neg_loglik(tp, data, p, q, slopes=slopes,
                                     **PLAIN_PK)) - f0) / step[i]
        return float(np.max(np.abs(g)))
    except Exception:
        return float("nan")


def _nelder_mead(theta0, data, p, q, maxiter, slopes=None):
    """Derivative-free descent on the scoring function itself.

    Last resort, and the only attempt that cannot be defeated by a line search
    on a different objective: the simplex keeps its best vertex, so the point it
    returns is never worse than ``theta0``. It is weak in this many dimensions,
    which is why it is last rather than first.
    """
    def obj(th):
        try:
            v = float(neg_loglik(np.asarray(th, float), data, p, q,
                                 slopes=slopes, **PLAIN_PK))
        except Exception:
            return 1e10
        return v if np.isfinite(v) else 1e10

    res = minimize(obj, np.asarray(theta0, float), method="Nelder-Mead",
                   options={"maxiter": maxiter, "fatol": 1e-6, "xatol": 1e-6,
                            "adaptive": True})
    return np.asarray(res.x, float), bool(res.success)


def finish_mle(data, p, q, theta_start, ll_start=None, seed=0, maxiter=1000,
               use_jax_grad=True, ma_above_ar=False, fit_kwargs=None,
               tol=1e-8, gtol_rel=1e-4, nm_maxiter=4000, attempts=None,
               precondition=None, slopes=None):
    """Optimize the ordinary likelihood from ``theta_start`` until it converges
    to a point at least as good as the start.

    An attempt is accepted when it satisfies both halves of what makes a point
    an MLE of the ordinary likelihood:

    * not worse than the start, ``ll >= ll_start - tol``; and
    * stationary, ``grad_norm <= gtol_rel * max(1, |ll|)``, on the ordinary
      likelihood's own gradient.

    Stationarity is tested directly rather than read off scipy's ``success``
    flag. BFGS reports ``success=False`` whenever its line search ends on
    precision loss, which is routine here and happens at points whose gradient
    sup-norm is order 1e-4; gating on the flag would push nearly every fit down
    to the derivative-free rung for no reason. The flag is still recorded in the
    trail, it just does not decide anything.

    ``precondition="curvature"`` reparameterizes the search by the curvature
    diagonal (see fit.curvature_scales). It only bites on the rungs that run the
    JAX objective, which is where the iteration budget is actually spent, and it
    changes nothing about the acceptance test: ``ll`` and ``grad_norm`` are both
    still measured on the ordinary likelihood in theta coordinates.

    ``fit_kwargs`` are the stage-2 settings production already uses; the first
    attempt runs them unchanged, later attempts override only what has to
    change. Returns a dict with the winning ``theta`` and ``ll``, the
    ``attempt`` that produced it, a ``converged`` flag, ``grad_norm`` and
    ``grad_norm_start`` (the ordinary likelihood's gradient sup-norm at the
    finish and at the starting point), and a ``trail`` listing every attempt
    with its ordinary log-likelihood, gradient norm and scipy flag.

    ``ll`` is always at least the ordinary likelihood at ``theta_start``,
    because the start stays in as the last candidate. What changes relative to
    the old rule is that the start is only *returned* when every attempt
    failed, and that case comes back ``converged=False`` rather than being
    called an MLE.

    ``slopes`` is part of the objective, not a preprocessing step. Under
    --slopes-in-likelihood the per-band trend is estimated jointly with
    everything else, so a likelihood taken without it is a different function.
    It may be passed explicitly or as a ``slopes`` key inside ``fit_kwargs``;
    the explicit argument wins, and either way the one value is used by every
    rung and by the scoring calls, so the ladder cannot end up comparing two
    different functions. It defaults to None because the simulation study has
    no trends.
    """
    theta_start = np.asarray(theta_start, float)

    # Resolve slopes before anything scores: the two spellings must collapse to
    # one value here, or a rung could be fit with trends and scored without.
    fit_kwargs = dict(fit_kwargs or {})
    _fk_slopes = fit_kwargs.pop("slopes", None)
    if slopes is None:
        slopes = _fk_slopes

    ll0 = plain_ll(theta_start, data, p, q, slopes=slopes) if ll_start is None \
        else float(ll_start)

    # The ordinary likelihood's gradient at the PMLE, recorded next to the one
    # at the finish. This is the direct measurement behind retrying rather than
    # falling back: the stage-1 point was optimized against a different
    # function, so there is no reason for this number to be small, and where it
    # is large the PMLE is demonstrably not a stationary point of the likelihood
    # the estimate would be reported on.
    gnorm0 = plain_grad_norm(theta_start, data, p, q, use_jax=use_jax_grad,
                             slopes=slopes)

    # Build the curvature scales ONCE, here, rather than letting each rung
    # rebuild them. Every rung starts from theta_start and every rung runs the
    # same plain objective (PLAIN_PK), so the scales are identical across the
    # ladder and the Hessian is the expensive part. Failure is not fatal: it
    # just leaves the rungs in raw coordinates, which is the historical search.
    if isinstance(precondition, str) and precondition == "curvature":
        try:
            precondition = curvature_scales_at(data, p, q, theta_start,
                                               slopes=slopes, **PLAIN_PK)
        except Exception as exc:
            _log.warning(f"[stage2] preconditioner unavailable ({exc}); "
                  f"rungs run in raw coordinates")
            precondition = None

    base = dict(fit_kwargs)
    for k in ("warm_theta", "local_only", "use_jax_grad", "seed", "maxiter",
              "ma_above_ar", "diag_load_lambda", "restart_score",
              "precondition"):
        base.pop(k, None)
    n_restarts_multi = int(base.pop("n_restarts", 1) or 1)
    if slopes is not None:
        # Forwarded to every fit() rung from the one resolved value, so the
        # rungs optimize the same function the scoring calls evaluate.
        base["slopes"] = slopes

    trail = []
    best = {"theta": theta_start, "ll": ll0, "attempt": "start",
            "grad_norm": float("nan"), "ok": False, "res": None}

    # `attempts=()` means run no rungs and report the start, which is the
    # fallback outcome without an optimizer in the way. It is how the
    # fallback contract is tested; `attempts=None` still means the full
    # ladder, so production is unaffected.
    for name in (LADDER if attempts is None else attempts):
        res = None
        if name == "nelder_mead":
            try:
                th, ok = _nelder_mead(theta_start, data, p, q, nm_maxiter,
                                      slopes=slopes)
                # The simplex escapes, then a local BFGS from where it landed
                # certifies the point. Two reasons the rung does not stop at the
                # simplex: Nelder-Mead terminates on a small SIMPLEX, which says
                # nothing about the gradient, so its point often fails the
                # stationarity test that decides acceptance; and it returns no
                # curvature, while the SE dispatcher reads hess_inv off the
                # fit() result. Keeping the BFGS point only when it does not
                # fall back means the rung is still monotone.
                pol = fit(data, p, q, n_restarts=1, warm_theta=th,
                          maxiter=maxiter, seed=seed, ma_above_ar=ma_above_ar,
                          use_jax_grad=False, local_only=True,
                          **dict(base, **PLAIN_PK))
                th_pol = None if pol is None else pol.get("theta")
                keep = (th_pol is not None and
                        plain_ll(th_pol, data, p, q, slopes=slopes)
                        >= plain_ll(th, data, p, q, slopes=slopes) - tol)
                if keep:
                    th, ok, res = th_pol, bool(pol.get("success", ok)), pol
            except Exception as exc:
                trail.append({"attempt": name, "ll": float("nan"),
                              "error": str(exc)})
                continue
        else:
            local = name.endswith("_local")
            kw = dict(base)
            kw["use_jax_grad"] = bool(use_jax_grad) and name.startswith("jax")
            kw["local_only"] = local
            kw.update(PLAIN_PK)
            if not local:
                # The multi-seed rung is the one that can hand back another
                # basin's optimum. With local_only=False, fit() appends the
                # damping-reset seeds for p >= 2, and under JAX it ranks those
                # seeds on the objective it minimized rather than on the
                # likelihood this ladder scores them with, so the seed it
                # returns need not be the best one it found and the losers are
                # discarded unscored. Ranking on plain_ll removes that: the same
                # function decides the winner inside fit() and the acceptance
                # test outside it. The local rungs run a single seed, so ranking
                # cannot change their outcome and they are left exactly as
                # production ran them.
                kw["restart_score"] = \
                    lambda th: plain_ll(th, data, p, q, slopes=slopes)
            try:
                res = fit(data, p, q,
                          n_restarts=(1 if local else n_restarts_multi),
                          warm_theta=theta_start, maxiter=maxiter, seed=seed,
                          ma_above_ar=ma_above_ar,
                          precondition=precondition, **kw)
            except Exception as exc:
                trail.append({"attempt": name, "ll": float("nan"),
                              "error": str(exc)})
                continue
            th = None if res is None else res.get("theta")
            ok = bool(res.get("success", False)) if res else False
            if th is None:
                trail.append({"attempt": name, "ll": float("nan"),
                              "error": "no theta"})
                continue

        # Scored by the same call every time; the optimizer's own number never
        # enters the comparison.
        ll = plain_ll(th, data, p, q, slopes=slopes)
        gnorm = (plain_grad_norm(th, data, p, q, use_jax=use_jax_grad,
                                 slopes=slopes)
                 if np.isfinite(ll) else float("nan"))
        rose = np.isfinite(ll) and (not np.isfinite(ll0) or ll >= ll0 - tol)
        stationary = np.isfinite(gnorm) and \
            gnorm <= gtol_rel * max(1.0, abs(ll))
        trail.append({"attempt": name, "ll": ll, "grad_norm": gnorm,
                      "scipy_success": bool(ok), "rose": bool(rose),
                      "stationary": bool(stationary)})

        cand = {"theta": np.asarray(th, float), "ll": ll, "attempt": name,
                "grad_norm": gnorm, "ok": bool(rose and stationary),
                "res": res}
        # Prefer a point that qualifies as an MLE over one that merely scores
        # higher, so a non-stationary excursion cannot displace a converged
        # optimum. Among equals, the higher likelihood wins.
        if np.isfinite(ll) and (
                (cand["ok"] and not best["ok"])
                or (cand["ok"] == best["ok"] and ll > best["ll"] + tol)):
            best = cand
        if cand["ok"]:
            # Rose above its start and is stationary for the ordinary
            # likelihood: an MLE reached from the regularized initialization.
            break

    return {"theta": best["theta"], "ll": best["ll"],
            "attempt": best["attempt"], "converged": bool(best["ok"]),
            "ll_start": ll0, "grad_norm": best["grad_norm"],
            "grad_norm_start": gnorm0, "trail": trail,
            # The winning fit()'s own result dict, when a fit() rung won, so a
            # caller can read mu and the state-space matrices off the SAME run
            # that produced theta. None for the Nelder-Mead rung and for a
            # ladder that produced nothing.
            "fit_result": best["res"]}
