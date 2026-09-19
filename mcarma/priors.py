"""Canonical prior constructors shared by every MCARMA driver.

Historically each driver (fits/process_quasars/process_quasars.py, the sim study
fits/21sim/sim_study.py, and the LSST-downsample fits/21sim/fit_lsst_downsample.py)
carried its OWN inline copy of the Sigma soft-box construction. That is exactly
the kind of duplication that let a factor-of-2 floor bug live in one copy while the
others were fixed (see the memory node mv-coverage-fix-options). This module is the
single source of truth; the drivers import from here.

Only the pure Sigma soft-box lives here. The AR/MA-band and AR-damping penalties are
independent priors the caller merges into the same ``prior_kwargs`` dict; keeping them
out of ``sigma_softbox`` preserves the exact per-driver merge each currently performs
(and therefore byte-identical numbers -- see tests/test_priors_golden.py)."""
import numpy as np


def sigma_softbox(y, band, d, lam, ceiling_mult, onesided=True, floor_mult=1e-4):
    """Per-band Sigma soft-box prior kwargs from the (centered) data.

    Data-driven one-sided log-normal soft-box on the Cholesky log-diagonal of the
    process-noise covariance Sigma. Floor center ``c_b = 2*log(floor_mult*sd_b)``,
    ceiling center ``2*log(ceiling_mult*sd_b)``; the penalty is flat between them.
    ``sd_b`` is the per-band empirical std of the (centered) series ``y``, with a
    global-scatter fallback for bands with <2 points so ``c_b`` is always finite.

    floor_mult was historically 0.1, which floors the LATENT driving variance at
    (0.1*sd_obs)^2 -- conflating driving variance with OBSERVED variance. For a slow
    AR the observed (stationary) variance is the driving variance amplified by the
    integration gain, so the true driving variance is orders of magnitude smaller; a
    0.1 floor then penalized slow truths by thousands of nats and drove the (2,1)
    collapse. floor_mult=1e-4 keeps only an anti-Sigma->0 numerical guard while
    admitting slow truths (see memory mcarma-21-collapse-rootcause).

    Returns only the ``chol_ridge_*`` keys (``{}`` when ``lam<=0``); the caller merges
    in ``ar_band_lambda``/``ma_band_lambda``/``ar_damping_lambda`` etc. as before.
    """
    if lam <= 0.0:
        return {}
    global_sd = float(np.std(y)) if y.size else 1.0
    center = np.empty(d)
    center_hi = np.empty(d)
    for b in range(d):
        yb = y[band == b]
        sd_b = float(np.std(yb)) if yb.size >= 2 else global_sd
        if not (sd_b > 0.0):
            sd_b = global_sd if global_sd > 0.0 else 1.0
        center[b] = 2.0 * np.log(floor_mult * sd_b)
        center_hi[b] = 2.0 * np.log(ceiling_mult * sd_b)
    pk = dict(chol_ridge_lambda=lam, chol_ridge_center=center,
              chol_ridge_onesided=onesided)
    if onesided and ceiling_mult > 0.0:
        pk["chol_ridge_center_hi"] = center_hi
    return pk
