"""Shared pieces for the mcarma examples.

Three things all the examples need: the SDSS Stripe 82 file format, the penalty
stack the production pipeline runs with, and the AICc parameter count. They live
here so the examples stay short and so a change to any of them is made once.

Nothing in here is part of the mcarma public API. It is example scaffolding, and
the file format in particular is specific to the bundled data.
"""
import os

import numpy as np

# --- SDSS Stripe 82 file format (17-column ugriz table) --------------------
N_BANDS = 5
BAND_LABELS = ["u", "g", "r", "i", "z"]
BAND_COLORS = {"u": "tab:purple", "g": "tab:green", "r": "tab:red",
               "i": "tab:brown", "z": "tab:gray"}
MAG_MIN, MAG_MAX = 10.0, 30.0
MISSING_MAG = -99.99

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OBJ = os.path.join(HERE, "data", "1000679")


def load_stripe82(filepath):
    """Parse a 17-column SDSS Stripe 82 quasar file into (t, y, band, R).

    Each row holds (t, mag, err) triplets for the 5 ugriz bands (15 columns)
    plus (RA, Dec). Missing/invalid epochs (sentinel magnitude, non-positive
    error, out-of-range magnitude) are dropped. Returns time-sorted arrays;
    R is the measurement *variance* (err**2).
    """
    raw = np.loadtxt(filepath)
    if raw.ndim == 1:
        raw = raw[np.newaxis, :]
    assert raw.shape[1] == 17, f"{filepath}: expected 17 columns, got {raw.shape[1]}"

    t, y, band, R = [], [], [], []
    for row in raw:
        for b in range(N_BANDS):
            tv, mag, err = row[b * 3], row[b * 3 + 1], row[b * 3 + 2]
            if abs(mag - MISSING_MAG) < 0.01 or err <= 0 or not (MAG_MIN <= mag <= MAG_MAX):
                continue
            t.append(tv); y.append(mag); band.append(b); R.append(err ** 2)
    if not t:
        raise ValueError(f"{filepath}: no valid observations")
    t, y, band, R = map(np.asarray, (t, y, band, R))
    order = np.argsort(t, kind="stable")
    return t[order], y[order].astype(float), band[order].astype(int), R[order]


def production_prior_kwargs(y, band, d):
    """The production penalty stack (matches process_quasars.py defaults).

    Sigma soft-box: one-sided log-normal on the driving variance, floor at
    (1e-4 * band_std)^2 and ceiling at (10 * band_std)^2, lambda=10. Plus the
    AR/MA band and AR-damping log-uniform guardrails (lambda=1e3).
    """
    global_sd = float(np.std(y)) if y.size else 1.0
    center_lo = np.empty(d)
    center_hi = np.empty(d)
    for b in range(d):
        yb = y[band == b]
        sd_b = float(np.std(yb)) if yb.size >= 2 else global_sd
        if not (sd_b > 0.0):
            sd_b = global_sd if global_sd > 0.0 else 1.0
        center_lo[b] = 2.0 * np.log(1e-4 * sd_b)   # floor on latent driving var
        center_hi[b] = 2.0 * np.log(10.0 * sd_b)   # ceiling walls off Sigma->inf
    return dict(
        chol_ridge_lambda=10.0,
        chol_ridge_center=center_lo,
        chol_ridge_onesided=True,
        chol_ridge_center_hi=center_hi,
        ar_band_lambda=1e3,
        ma_band_lambda=1e3,
        ar_damping_lambda=1e3,
    )


def n_params(d, p, q):
    """Free-parameter count for AICc: AR + MA + Cholesky(Sigma) + per-band mean."""
    return d * p + d * q + d * (d + 1) // 2 + d
