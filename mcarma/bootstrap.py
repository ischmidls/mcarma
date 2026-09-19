"""Per-object parametric bootstrap: SE for every parameter + finite-sample
bias-correction of the MA coefficient b.

See docs/METHODS.Rmd sec 8-9 (bias-correction b_bc = 2*b_hat - boot_mean; subsampling SE).

Why this exists.  The MLE of a moving-average coefficient carries a known
O(1/n) downward finite-sample bias (Cox & Snell 1968 for the general MLE
expansion; Cordeiro & Klein 1994 for ARMA specifically; the classic MA(1)
"pile-up" toward the non-invertibility boundary).  The simulation study measured
it at ~22% (median -0.245 in log b) and a lambda-probe confirmed it is
estimator-intrinsic, NOT our prior (it persists, and grows, as the MA band prior
lambda -> 0).  The SEs are well-calibrated (bootstrap SE ~ Hessian SE); only the
point estimate is offset.

Crucially the correction is estimated PER OBJECT, not applied as a blanket shift
from the study's (narrow) parameter region: for each object we simulate from its
OWN fit on its OWN cadence, refit with the same per-band warm start, and take the
empirical mean/SD over refits.  bias-corrected theta = 2*theta_fit - boot_mean.

Mirrors the validated real-data bootstrap in fits/process_quasars/validate_object.py
(object 1035792).  Depends only on the mcarma package (warm start, simulate, fit,
per-band centering), so it lives here in the core library.
"""
import logging
import warnings
import numpy as np

from mcarma.preprocess import center_bands
from mcarma.model_utils import build_perband_warm_theta
from mcarma.observation import ObservationData
from mcarma.simulate import simulate
from mcarma.fit import fit

_log = logging.getLogger(__name__)


def parametric_bootstrap_se(theta_fit, data, p, q, prior=None, n_boot=24,
                            n_restarts_perband=4, maxiter=300,
                            use_jax_grad=False, seed0=7000, verbose=False):
    """Parametric bootstrap of the fit at (p, q) on `data`'s exact design.

    Simulate `n_boot` CARMA(p,q) light curves from `theta_fit` on the same
    times / bands / per-obs noise as `data`, refit each at (p,q) with a per-band
    warm start (no slopes: the simulated process is trend-free), and summarise
    the refit theta vectors.

    Returns dict(theta_se, boot_mean, theta_bc, boot_n, boot_thetas) or None if
    fewer than 2 reps succeeded.  theta_bc = 2*theta_fit - boot_mean is the
    bias-corrected estimate; theta_se is the per-parameter empirical SD (the
    honest, Hessian-free SE).  For q >= 1 the MA coeff is theta[p*d].
    """
    theta_fit = np.asarray(theta_fit, dtype=float)
    d = data.d
    t = data.t_obs
    band = data.band
    R = np.array([data.R_list[k][0, 0] for k in range(data.n)])
    tmpl = ObservationData(t, np.zeros(data.n), band, R, d=d)
    prior = prior or {}

    thetas = []
    for m in range(n_boot):
        rng = np.random.default_rng(seed0 + m)
        try:
            sim = simulate(theta_fit, tmpl, p=p, q=q,
                           initial_state="stationary",
                           seed=int(rng.integers(1 << 31)))
            yb = center_bands(t, sim.data.y_obs, band)
            db = ObservationData(t, yb, band, R, d=d)
            warm = build_perband_warm_theta(
                t, yb, band, R, d=d, p=p, q=q,
                n_restarts=n_restarts_perband,
                seed=int(rng.integers(1 << 31)))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r = fit(db, p, q, n_restarts=1, warm_theta=warm,
                        maxiter=maxiter, use_jax_grad=use_jax_grad, **prior)
            th = np.asarray(r["theta"], dtype=float)
            if th.shape != theta_fit.shape:
                continue
            thetas.append(th)
            if verbose:
                bstr = f" b11_log={th[p * d]:+.3f}" if q >= 1 else ""
                _log.info(f"    [boot {m:2d}]{bstr}")
        except Exception as exc:
            _log.warning(f"    [boot {m:2d}] failed: {exc}")
            continue

    if len(thetas) < 2:
        return None
    thetas = np.vstack(thetas)
    theta_se = np.std(thetas, axis=0, ddof=1)
    boot_mean = np.mean(thetas, axis=0)
    theta_bc = 2.0 * theta_fit - boot_mean
    return dict(theta_se=theta_se, boot_mean=boot_mean, theta_bc=theta_bc,
                boot_n=int(thetas.shape[0]), boot_thetas=thetas)


def subsampling_se(theta_fit, data, p, q, prior=None, m=None, n_sub=40,
                   n_restarts_perband=4, maxiter=300, use_jax_grad=False,
                   seed0=9000, verbose=False):
    """m-out-of-n subsampling SE (Politis-Romano-Wolf), WITHOUT replacement.

    Independent of the parametric bootstrap above: instead of regenerating data
    from the fitted model, we refit on genuine subsamples of the observed light
    curve, so it does NOT assume the model is correctly specified.

    Procedure.  Draw ``n_sub`` subsamples of ``m = round(n**(2/3))`` epochs
    without replacement, stratified by band (so every band keeps its share and
    stays identifiable), refit each at (p, q) with the same per-band warm start
    and prior, take the per-parameter SD of the subsample estimates, and rescale
    to the full-sample SE by ``sqrt(m/n)``.  The rescaling is the subsampling
    identity: if ``sqrt(k)*(theta_k - theta)`` shares one limiting law, then
    ``SD(theta_m) ~ sqrt(V/m)`` and the full-sample SE is
    ``SD(theta_m)*sqrt(m/n) ~ sqrt(V/n)``.  CARMA is continuous-time, so dropping
    epochs just yields a sparser irregular series the Kalman filter handles
    natively.

    Returns dict(theta_se, m, n, sub_n, theta_sub_std, sub_thetas) or None if
    fewer than 2 subsample fits succeeded.  For q >= 1 the MA coeff is
    theta[p*d].
    """
    theta_fit = np.asarray(theta_fit, dtype=float)
    d = data.d
    t = np.asarray(data.t_obs, dtype=float)
    band = np.asarray(data.band, dtype=int)
    y = np.asarray(data.y_obs, dtype=float)
    R = np.array([data.R_list[k][0, 0] for k in range(data.n)])
    n = int(data.n)
    prior = prior or {}
    if m is None:
        m = int(round(n ** (2.0 / 3.0)))
    m = int(max(d + 2, min(m, n - 1)))

    band_idx = [np.where(band == b)[0] for b in range(d)]
    thetas = []
    for s in range(n_sub):
        rng = np.random.default_rng(seed0 + s)
        parts = []
        for b in range(d):
            bi = band_idx[b]
            if len(bi) == 0:
                continue
            mb = int(round(len(bi) * m / n))
            mb = min(max(mb, 1), len(bi))
            parts.append(rng.choice(bi, size=mb, replace=False))
        if not parts:
            continue
        idx = np.sort(np.concatenate(parts))
        try:
            ys = center_bands(t[idx], y[idx], band[idx])
            ds = ObservationData(t[idx], ys, band[idx], R[idx], d=d)
            warm = build_perband_warm_theta(
                t[idx], ys, band[idx], R[idx], d=d, p=p, q=q,
                n_restarts=n_restarts_perband,
                seed=int(rng.integers(1 << 31)))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r = fit(ds, p, q, n_restarts=1, warm_theta=warm,
                        maxiter=maxiter, use_jax_grad=use_jax_grad, **prior)
            th = np.asarray(r["theta"], dtype=float)
            if th.shape != theta_fit.shape:
                continue
            thetas.append(th)
            if verbose:
                bstr = f" b11_log={th[p * d]:+.3f}" if q >= 1 else ""
                _log.info(f"    [sub {s:2d}] m={idx.size}{bstr}")
        except Exception as exc:
            _log.warning(f"    [sub {s:2d}] failed: {exc}")
            continue

    if len(thetas) < 2:
        return None
    thetas = np.vstack(thetas)
    theta_sub_std = np.std(thetas, axis=0, ddof=1)
    theta_se = theta_sub_std * np.sqrt(m / n)
    return dict(theta_se=theta_se, m=int(m), n=int(n),
                sub_n=int(thetas.shape[0]), theta_sub_std=theta_sub_std,
                sub_thetas=thetas)
