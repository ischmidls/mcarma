"""
Utilities for MCARMA parameter transformations and state-space construction.

See docs/METHODS.Rmd sec 1-2 (state-space F/G/H, Jones parameterization).

This module implements:

- packing/unpacking of unconstrained optimization vectors,
- stable AR and minimum-phase MA factor parameterizations,
- Cholesky covariance parameterization,
- continuous-time state-space realization construction.

Parameterization
----------------
Positive AR and MA factor coefficients are represented in log-space within
the optimization vector. Cholesky diagonal entries are likewise stored in
log-space to guarantee positive definiteness of the latent driving covariance.

The unconstrained optimization vector is ordered as

    [AR factors | MA factors | Cholesky entries | mu]

where:

- linear factors store log(a),
- quadratic factors store log(a1), log(a0),
- Cholesky diagonals store log(L_ii^2),
- Cholesky off-diagonals are stored directly.

Exponentiation during unpacking guarantees positivity of all constrained
parameters while permitting unconstrained optimization in R^n.
"""

import warnings
import numpy as np
from .parameters import expand_jones_to_coeffs


def unpack_params_jones(theta, d, p, q):
    """
    Unpack flat theta vector into Jones AR/MA factor lists and Sigma.

    AR/MA Jones factor coefficients are stored as their logarithms in theta.
    exp() is applied here to recover the raw positive coefficients before
    building factor tuples.  The Cholesky diagonal is similarly stored as
    log(L_ii^2) and scaled by 0.5 after exp() here; off-diagonal entries are stored raw.

    Parameters
    ----------
    theta : np.ndarray
        Flat vector [log-AR | log-MA | Chol entries]
        (mu is NOT part of theta_carma; caller strips it before passing)
    d : int, number of bands
    p : int, AR order
    q : int, MA order

    Returns
    -------
    ar_factors : list[d] of list of a tuples  (raw positive values)
    ma_factors : list[d] of list of b tuples   (raw positive values)
    V          : (d, d) positive-definite latent driving covariance
    """
    idx = 0
    ar_factors = []
    ma_factors = []

    has_linear_ar = p % 2
    num_quads_ar  = p // 2
    for b in range(d):
        factors = []
        if has_linear_ar:
            factors.append((np.exp(theta[idx]),))
            idx += 1
        for _ in range(num_quads_ar):
            factors.append((np.exp(theta[idx]),
                            np.exp(theta[idx + 1])))
            idx += 2
        ar_factors.append(factors)

    has_linear_ma = q % 2
    num_quads_ma  = q // 2
    for b in range(d):
        factors = []
        if has_linear_ma:
            factors.append((np.exp(theta[idx]),))
            idx += 1
        for _ in range(num_quads_ma):
            factors.append((np.exp(theta[idx]),
                            np.exp(theta[idx + 1])))
            idx += 2
        ma_factors.append(factors)

    # Cholesky: diagonal stored as log(L_ii^2), off-diagonal stored raw.
    n_chol     = d * (d + 1) // 2
    L          = np.zeros((d, d))
    rows, cols = np.tril_indices(d)
    for k in range(n_chol):
        i, j = int(rows[k]), int(cols[k])
        if i == j:
            L[i, j] = np.exp(0.5 * theta[idx + k])   # log(L_ii^2) → L_ii > 0
        else:
            L[i, j] = theta[idx + k]            # off-diagonal: raw
    V = L @ L.T
    idx  += n_chol

    return ar_factors, ma_factors, V


def pack_params_jones(ar_factors, ma_factors, V):
    """
    Pack AR/MA factor lists and V into a flat theta vector.

    AR/MA Jones factor coefficients are log-transformed before packing.
    The Cholesky diagonal is stored as log(L_ii^2).  Off-diagonal entries
    are stored raw.

    Parameters
    ----------
    ar_factors : list[d] of list of a tuples  (raw positive values)
    ma_factors : list[d] of list of b tuples   (raw positive values)
    V          : (d, d) positive-definite latent driving covariance

    Returns
    -------
    theta : np.ndarray  [log-AR | log-MA | Chol entries]
    """
    theta = []

    for factors in ar_factors:
        for f in factors:
            for v in f:
                theta.append(np.log(float(v)))

    for factors in ma_factors:
        for f in factors:
            for v in f:
                theta.append(np.log(float(v)))

    d = V.shape[0]
    L = np.linalg.cholesky(V)
    rows, cols = np.tril_indices(d)
    for i, j in zip(rows, cols):
        if i == j:
            theta.append(2.0 * np.log(float(L[i, j])))   # log(L_ii^2)
        else:
            theta.append(float(L[i, j]))            # off-diagonal raw

    return np.array(theta)


def build_state_space(ar_factors, ma_factors, V, d, p, q):
    """
    Construct continuous-time state-space matrices (F, G, H, V)
    from Jones AR/MA factor lists.

    Operates on raw (already-exponentiated) factor tuples as produced by
    unpack_params_jones().  Unchanged from previous version.

    Parameters
    ----------
    ar_factors : list[d] of list of a tuples  (raw positive values)
    ma_factors : list[d] of list of b tuples   (raw positive values)
    V : (d, d)
    d, p, q : int

    Returns
    -------
    F : (d*p, d*p)
    G : (d*p, d)
    H : (d, d*p)
    V : (d, d)
    """
    ar_coefs = np.zeros((d, p))
    for b in range(d):
        coeffs = expand_jones_to_coeffs(ar_factors[b], p)
        ar_coefs[b, :] = coeffs[1:][::-1]

    ma_coefs = np.zeros((d, q)) if q > 0 else np.zeros((d, 0))
    for b in range(d):
        if q > 0:
            coeffs = expand_jones_to_coeffs(ma_factors[b], q)
            ma_coefs[b, :] = coeffs[1:][::-1]

    F = np.zeros((d * p, d * p))
    for i in range(p - 1):
        F[i * d:(i+1) * d, (i+1) * d:(i+2) * d] = np.eye(d)
    for lag in range(p):
        F[(p-1) * d:p * d, lag * d:(lag+1) * d] = -np.diag(ar_coefs[:, lag])

    G = np.zeros((d * p, d))
    G[(p-1) * d:p * d, :] = np.eye(d)

    H = np.zeros((d, d * p))
    H[:, 0:d] = np.eye(d)
    for lag in range(q):
        H[:, (lag+1) * d:(lag+2) * d] = np.diag(ma_coefs[:, lag])

    return F, G, H, V


def project_pd(M, min_eig=1e-9):
    """
    Project a symmetric matrix onto the positive-definite cone.

    Symmetrises M, clamps all eigenvalues to at least min_eig.
    Returns a diagonal matrix of min_eig if M contains non-finite values.
    """
    if not np.isfinite(M).all():
        return np.eye(M.shape[0]) * min_eig
    M = 0.5 * (M + M.T)
    eigvals, eigvecs = np.linalg.eigh(M)
    eigvals = np.maximum(eigvals, min_eig)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T

def _ar_roots_from_factor(factor: tuple) -> list:
    """
    Complex roots of one AR factor.
      (a_0,)       → monic s + a_0,       root = −a_0  (real)
      (a_1, a_0)   → monic s²+a_1·s+a_0,  roots = −a_1/2 ± i·√(a_0−a_1²/4)
    """
    if len(factor) == 1:
        a0, = factor
        return [complex(-a0, 0.0)]
    elif len(factor) == 2:
        a1, a0 = factor
        disc = a1**2 - 4.0 * a0
        if disc >= 0:
            r = np.sqrt(disc)
            return [complex(-a1/2 + r/2, 0), complex(-a1/2 - r/2, 0)]
        else:
            r = np.sqrt(-disc)
            return [complex(-a1/2,  r/2), complex(-a1/2, -r/2)]
    else:
        raise ValueError(f"AR factor has unexpected length {len(factor)}")
 
 
def _ma_roots_from_factor(factor: tuple) -> list:
    """
    Complex roots of one MA factor.
      (b_0,)       → NON-monic 1 + b_0·s,         root = −1/b_0  (real)
      (b_1, b_0)   → NON-monic 1+b_1·s+b_0·s²,
                      roots of b_0·s²+b_1·s+1 = 0
                      = (−b_1 ± √(b_1²−4b_0)) / (2b_0)
    """
    if len(factor) == 1:
        b0, = factor
        return [complex(-1.0 / b0, 0.0)]
    elif len(factor) == 2:
        b1, b0 = factor
        disc = b1**2 - 4.0 * b0
        if disc >= 0:
            r = np.sqrt(disc)
            return [complex((-b1 + r) / (2*b0), 0),
                    complex((-b1 - r) / (2*b0), 0)]
        else:
            r = np.sqrt(-disc)
            return [complex(-b1/(2*b0),  r/(2*b0)),
                    complex(-b1/(2*b0), -r/(2*b0))]
    else:
        raise ValueError(f"MA factor has unexpected length {len(factor)}")


def min_polezero_dist_band(ar_factors_band: list, ma_factors_band: list) -> float:
    """
    Minimum Euclidean |AR_root − MA_root| over all pairs for one band.
 
    Replaces the version in mcarma_jones_fit_comparison.py that called
    mcarma.simulate.min_polezero_dist with wrong MA factor interpretation.
 
    ar_factors_band : list of factor tuples from unpack_params_jones (raw, post-exp)
    ma_factors_band : same for MA
    """
    ar_roots = []
    for f in ar_factors_band:
        ar_roots.extend(_ar_roots_from_factor(f))
    ma_roots = []
    for f in ma_factors_band:
        ma_roots.extend(_ma_roots_from_factor(f))
 
    if not ar_roots or not ma_roots:
        return float('inf')
 
    dists = [abs(ar - ma) for ar in ar_roots for ma in ma_roots]
    return float(min(dists))
