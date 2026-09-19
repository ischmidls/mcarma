"""Golden regression: mcarma.priors.sigma_softbox must reproduce, bit-for-bit, the
three legacy inline Sigma soft-box constructions it replaces --

  * fits/process_quasars/process_quasars.py:312-342   (real data; onesided default False)
  * fits/21sim/sim_study.py:136-169 build_sigma_softbox (bigindep sim; onesided True)
  * fits/21sim/fit_lsst_downsample.py:277-296          (LSST; onesided hardcoded True)

The legacy bodies are frozen below as *_legacy references. If sigma_softbox ever drifts
from any of them, this fails. Run: PYTHONUTF8=1 python -m pytest tests/test_priors_golden.py
(or plain `python tests/test_priors_golden.py`)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mcarma.priors import sigma_softbox  # noqa: E402


# ---- frozen legacy references (verbatim math from each driver) --------------

def _legacy_sim(y_c, band, d, lam, ceiling_mult, onesided=True, floor_mult=1e-4):
    """sim_study.build_sigma_softbox, frozen."""
    if lam <= 0.0:
        return {}
    global_sd = float(np.std(y_c)) if y_c.size else 1.0
    center = np.empty(d)
    center_hi = np.empty(d)
    for b in range(d):
        yb = y_c[band == b]
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


def _legacy_real(y, band, d, sigma_prior_lambda, sigma_prior_ceiling_mult,
                 sigma_prior_onesided, sigma_prior_floor_mult=1e-4):
    """process_quasars.py sigma-prior block, frozen. Returns ONLY the chol_ridge_*
    keys (the caller merges ar_band/ma_band separately, so those are excluded here)."""
    if not (sigma_prior_lambda > 0.0):
        return {}
    global_sd = float(np.std(y)) if y.size else 1.0
    sigma_center = np.empty(d, dtype=float)
    sigma_center_hi = np.empty(d, dtype=float)
    for b in range(d):
        yb = y[band == b]
        sd_b = float(np.std(yb)) if yb.size >= 2 else global_sd
        if not (sd_b > 0.0):
            sd_b = global_sd if global_sd > 0.0 else 1.0
        sigma_center[b] = 2.0 * np.log(sigma_prior_floor_mult * sd_b)
        sigma_center_hi[b] = 2.0 * np.log(sigma_prior_ceiling_mult * sd_b)
    prior_kwargs = dict(
        chol_ridge_lambda=sigma_prior_lambda,
        chol_ridge_center=sigma_center,
        chol_ridge_onesided=sigma_prior_onesided,
    )
    if sigma_prior_onesided and sigma_prior_ceiling_mult > 0.0:
        prior_kwargs["chol_ridge_center_hi"] = sigma_center_hi
    return prior_kwargs


def _legacy_lsst(y_centered, band, D, sigma_prior_lambda, sigma_prior_ceiling_mult):
    """fit_lsst_downsample.py sigma-prior block, frozen (onesided hardcoded True).
    Returns ONLY the chol_ridge_* keys."""
    global_sd = float(np.std(y_centered)) if y_centered.size else 1.0
    sigma_center = np.empty(D, dtype=float)
    sigma_center_hi = np.empty(D, dtype=float)
    for b in range(D):
        yb = y_centered[band == b]
        sd_b = float(np.std(yb)) if yb.size >= 2 else global_sd
        if not (sd_b > 0.0):
            sd_b = global_sd if global_sd > 0.0 else 1.0
        sigma_center[b] = 2.0 * np.log(1e-4 * sd_b)
        sigma_center_hi[b] = 2.0 * np.log(sigma_prior_ceiling_mult * sd_b)
    prior_kwargs = dict(
        chol_ridge_lambda=sigma_prior_lambda,
        chol_ridge_center=sigma_center,
        chol_ridge_onesided=True,
    )
    if sigma_prior_ceiling_mult > 0.0:
        prior_kwargs["chol_ridge_center_hi"] = sigma_center_hi
    return prior_kwargs


# ---- comparison helper ------------------------------------------------------

def _assert_pk_identical(a, b):
    assert set(a.keys()) == set(b.keys()), (sorted(a), sorted(b))
    for k in a:
        va, vb = a[k], b[k]
        if isinstance(va, np.ndarray):
            # byte-identical, not approx: same dtype, same bits.
            assert va.dtype == vb.dtype, (k, va.dtype, vb.dtype)
            assert np.array_equal(va, vb), (k, va, vb)
        else:
            assert va == vb, (k, va, vb)


def _rand_case(rng, d):
    n = int(rng.integers(30, 400))
    y = rng.normal(0.0, rng.uniform(0.05, 2.0), size=n)
    band = rng.integers(0, d, size=n)
    return y, band


def test_matches_sim_legacy():
    rng = np.random.default_rng(0)
    for _ in range(200):
        d = int(rng.integers(2, 7))
        y, band = _rand_case(rng, d)
        lam = float(rng.choice([0.0, 0.5, 1.0, 3.0]))
        ceil = float(rng.choice([0.0, 5.0, 10.0]))
        one = bool(rng.integers(0, 2))
        _assert_pk_identical(
            sigma_softbox(y, band, d, lam, ceil, onesided=one),
            _legacy_sim(y, band, d, lam, ceil, onesided=one),
        )


def test_matches_real_legacy():
    rng = np.random.default_rng(1)
    for _ in range(200):
        d = int(rng.integers(2, 7))
        y, band = _rand_case(rng, d)
        lam = float(rng.choice([0.0, 1.0, 2.0]))
        ceil = 10.0  # real default
        one = bool(rng.integers(0, 2))  # real default False, but sweep both
        got = sigma_softbox(y, band, d, lam, ceil, onesided=one, floor_mult=1e-4)
        ref = _legacy_real(y, band, d, lam, ceil, one, sigma_prior_floor_mult=1e-4)
        _assert_pk_identical(got, ref)


def test_matches_lsst_legacy():
    rng = np.random.default_rng(2)
    for _ in range(200):
        D = 6
        y, band = _rand_case(rng, D)
        lam = float(rng.choice([1.0, 2.0]))
        ceil = float(rng.choice([0.0, 10.0]))
        # LSST hardcodes onesided=True and floor_mult=1e-4.
        got = sigma_softbox(y, band, D, lam, ceil, onesided=True, floor_mult=1e-4)
        ref = _legacy_lsst(y, band, D, lam, ceil)
        _assert_pk_identical(got, ref)


def test_empty_lambda_returns_empty():
    rng = np.random.default_rng(3)
    y, band = _rand_case(rng, 5)
    assert sigma_softbox(y, band, 5, 0.0, 10.0) == {}
    assert sigma_softbox(y, band, 5, -1.0, 10.0) == {}


if __name__ == "__main__":
    test_matches_sim_legacy()
    test_matches_real_legacy()
    test_matches_lsst_legacy()
    test_empty_lambda_returns_empty()
    print("OK: sigma_softbox byte-identical to all 3 legacy inline constructions")
