"""The estimator the papers report, as one call.

`mcarma.fit` is this module's `fit`. It runs the two stages the papers describe:

1. Stage 1 maximizes a penalized likelihood. The penalties are a soft box on
   each band's driving variance, observable-band priors on the AR and MA roots,
   a floor on the AR decay rate, and diagonal loading of the driving
   covariance. They only serve to find a good starting point.
2. Stage 2 starts from the stage-1 point and maximizes the ordinary likelihood,
   with no penalty and no loading (`mcarma.stage2.finish_mle`).

The reported parameters, log-likelihood and AICc all come from stage 2.

The single-stage optimizer is still available as `mcarma.fit_penalized`, which
is `mcarma.fit.fit`. Called with its defaults it is NOT the paper's estimator:
with both `ar_band_lambda` and `ma_band_lambda` left as None it applies a
legacy near-hard hinge on the AR/MA timescales, kept so that runs made before
the soft priors existed still reproduce. Pass both as 0.0 to turn every band
penalty off.
"""
import numpy as np

from .fit import fit as fit_penalized
from .model_utils import compute_aicc, count_params, build_perband_warm_theta
from .optimizer_utils import build_state_space, unpack_params_jones
from .observation import ObservationData
from .preprocess import center_bands
from .priors import sigma_softbox
from .stage2 import finish_mle, PLAIN_PK

# The stage-1 settings the real-data and simulation runs were submitted with.
SIGMA_LAMBDA = 10.0
SIGMA_CEILING_MULT = 10.0
SIGMA_FLOOR_MULT = 1e-4
AR_BAND_LAMBDA = 1e3
MA_BAND_LAMBDA = 1e3
AR_DAMPING_LAMBDA = 1e3
DIAG_LOAD_LAMBDA = 0.05


def stage1_priors(y, band, d):
    """The stage-1 penalty weights for one data set.

    The driving-variance box is built from each band's scatter: floor at
    (1e-4 sd_b)^2, ceiling at (10 sd_b)^2. The other weights are fixed.
    """
    kw = sigma_softbox(y, band, d, SIGMA_LAMBDA, SIGMA_CEILING_MULT,
                       onesided=True, floor_mult=SIGMA_FLOOR_MULT)
    kw["ar_band_lambda"] = AR_BAND_LAMBDA
    kw["ma_band_lambda"] = MA_BAND_LAMBDA
    kw["ar_damping_lambda"] = AR_DAMPING_LAMBDA
    return kw


def fit(data, p, q=0, *, seed=0, n_restarts=6, n_restarts_perband=4,
        use_jax_grad=True, maxiter=1000, warm=True, slopes=None):
    """Fit MCARMA(p, q) with the two-stage estimator the papers report.

    Parameters
    ----------
    data : ObservationData
        Raw magnitudes are fine. Each band is centered before fitting, as in
        the papers, and the fitted means are shifted back, so ``mu`` and the
        mean entries of ``theta`` are on the scale of ``data.y_obs``.
    p, q : int
        AR and MA orders.
    seed : int
        Seeds the per-band warm start and both stages.
    n_restarts, n_restarts_perband : int
        Stage-1 restarts for the joint fit and for each per-band warm-start fit.
    use_jax_grad : bool
        Use the JAX analytic gradient where the order supports it.
    warm : bool
        Start stage 1 from independent per-band fits.
    slopes : ndarray (d,) or None
        Known per-band linear trends, modeled inside the likelihood.

    Returns
    -------
    dict
        ``theta``, ``F``, ``G``, ``H``, ``Sigma``, ``mu``: the stage-2 estimate.
        ``loglik``: the ordinary log-likelihood there. ``loglik_pure`` is the
        same number, under the key the single-stage result uses. ``k`` and
        ``aicc``. ``converged``: stage 2 rose above its start and ended
        stationary on the ordinary likelihood; when no attempt managed that,
        the stage-1 point is returned rescored on the ordinary likelihood and
        this is False. ``grad_norm``, ``attempt`` and ``trail`` describe the
        stage-2 search, ``hess_inv`` is the inverse Hessian from the winning
        attempt when it has one, ``loglik_stage1`` is the ordinary
        log-likelihood at the stage-1 point and ``stage1`` its full result
        (fitted to the centered data, so its means are near zero).
    """
    d = data.d
    t = np.asarray(data.t_obs, float)
    band = np.asarray(data.band, int)
    R = np.asarray(data.R_list, float).reshape(-1)
    y = center_bands(t, np.asarray(data.y_obs, float), band)
    offset = np.zeros(d)
    for b in np.unique(band):
        offset[b] = float(np.mean(np.asarray(data.y_obs, float)[band == b]))
    cdata = ObservationData(t, y, band, R, d)

    rng = np.random.default_rng(seed)
    warm_theta = None
    if warm:
        warm_theta = build_perband_warm_theta(
            t, y, band, R, d=d, p=p, q=q, n_restarts=n_restarts_perband,
            seed=int(rng.integers(1 << 31)), slopes=slopes)

    stage1 = fit_penalized(cdata, p, q, n_restarts=n_restarts,
                           warm_theta=warm_theta,
                           seed=int(rng.integers(1 << 31)),
                           use_jax_grad=use_jax_grad, maxiter=maxiter,
                           slopes=slopes, diag_load_lambda=DIAG_LOAD_LAMBDA,
                           **stage1_priors(y, band, d))

    fin = finish_mle(cdata, p, q, np.asarray(stage1["theta"], float),
                     seed=int(rng.integers(1 << 31)), maxiter=maxiter,
                     use_jax_grad=use_jax_grad, precondition="curvature",
                     slopes=slopes, fit_kwargs=dict(PLAIN_PK))

    # Rebuild the state space from the reported theta rather than reading it
    # off a rung's result, so it is always the unloaded model at that theta,
    # including when the stage-1 point is what comes back.
    theta = np.asarray(fin["theta"], float).copy()
    ar, ma, V = unpack_params_jones(theta[:-d], d, p, q)
    F, G, H, Sigma = build_state_space(ar, ma, V, d, p, q)
    theta[-d:] += offset
    ll = float(fin["ll"])
    k = count_params(d, p, q)
    res_fin = fin.get("fit_result") or {}
    return {
        "theta": theta,
        "F": F, "G": G, "H": H, "Sigma": Sigma,
        "mu": theta[-d:].reshape(d, 1),
        "loglik": ll,
        "loglik_pure": ll,
        "k": k,
        "aicc": float(compute_aicc(ll, k, data.n)),
        "converged": bool(fin["converged"]),
        "grad_norm": float(fin["grad_norm"]),
        "attempt": str(fin["attempt"]),
        "trail": fin["trail"],
        "hess_inv": res_fin.get("hess_inv"),
        "loglik_stage1": float(fin["ll_start"]),
        "stage1": stage1,
        "slopes": slopes,
    }
