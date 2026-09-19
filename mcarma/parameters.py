"""
mcarma/parameters.py
====================
Parameter transformations and polynomial expansions for MCARMA models.

Provides core functionality for handling AR and MA parameters in Jones factor
representation, including:
  1. positive_params() - Map unconstrained optimization variables to positive values
  2. expand_jones_to_coeffs() - Expand Jones factors to polynomial coefficients
  3. build_coefficient_matrices() - Build coefficient matrices for all bands
  4. matrix_companion_eigenvalues() - Compute eigenvalues of block companion matrix

This is a CONTINUOUS-TIME model (s-plane), so stability means all roots have
negative real part; there is no unit-circle / discrete-time constraint. The Jones
factor representation expresses AR and MA polynomials as products of linear and
quadratic factors so that constrained optimization becomes unconstrained
optimization over positive factor coefficients (stored in log space upstream).
An MA linear factor b defines the non-monic 1 + b*s with its zero at -1/b.

See docs/METHODS.Rmd sec 2 for the parameterization and conventions.

Parameter Representations
-------------------------
MCARMA models have three types of parameters, each transformed differently:

AR as (stability):
    Represented as Jones factors where each factor (z + a) or (z² + a₁z + a₂)
    corresponds to a root with negative real part (continuous-time stability).
    During optimization, raw parameters are unconstrained and transformed
    to ensure negative real parts via mapping functions (not in this module).

MA bs (invertibility):
    Represented as Jones factors similar to AR but ensuring roots inside
    unit circle (discrete-time) or with negative real parts (continuous-time).
    The expansion follows same convention but requires sign adjustments.

Cholesky L (positive definiteness):
    Lower-triangular Cholesky factor of innovation covariance V = LLᵀ.
    Diagonal entries are positive, off-diagonals are unconstrained.

Band means μ:
    Unconstrained location parameters.

Jones Factor Expansion Convention
---------------------------------
This module implements a specific expansion convention used throughout the
MCARMA pipeline:

AR convention (used internally):
    - Linear factor (a): expands to (z + a)
    - Quadratic factor (a₁, a₂): expands to (z² + a₁z + a₂)
    - Resulting polynomial is monic: [1, c_{order-1}, ..., c₀]
    - Highest power first (standard numpy polynomial convention)

For AR polynomials: This convention is natural and coefficients are used
directly in build_state_space() for constructing companion matrices.

For MA polynomials: The same expansion produces monic polynomials, but the
MCARMA model requires monic polynomials in lowest-power-first form with
constant term 1. Therefore, the result is reversed: coeffs[::-1] before use.

Do NOT use expand_jones_to_coeffs() on MA factors for diagnostics without
applying the same reversal, as roots will be incorrectly interpreted.

Mathematical Details
--------------------
Jones factor representation:
    For a polynomial A(s) = s^p + a₁s^{p-1} + ... + a_p with roots r₁,...,r_p,
    factor as A(s) = ∏_{j} (s + a_j) ∏_{k} (s² + a1_k s + a2_k)
    where a_j > 0 and a1_k, a2_k ensure complex conjugate pairs with negative
    real parts.

For continuous-time stability, each factor contributes:
    - Linear: root at -a (negative real)
    - Quadratic: roots with real part = -a1_k/2 (negative if a1_k > 0)

The expansion uses convolution to multiply factors sequentially, then
pads/truncates to achieve desired order.

Functions
---------
positive_params(raw)
    Map unconstrained raw parameters to strictly positive values via exp().
    Used for Cholesky diagonal entries and other positivity constraints.

expand_jones_to_coeffs(factors, order)
    Expand list of factor tuples into monic polynomial coefficients.
    
    Parameters:
        factors : list of tuples
            Each tuple is (a,) for linear or (a1, a2) for quadratic factors
        order : int
            Desired polynomial degree (p for AR, q for MA)
    
    Returns:
        coeffs : ndarray of length order+1
            [1, a_{order-1}, ..., a₀] (monic, highest power first)

build_coefficient_matrices(factors_list, order)
    Build (d, order) coefficient matrix for all bands simultaneously.
    
    Parameters:
        factors_list : list of length d
            factors_list[b] is the factor list for band b
        order : int
            Polynomial order (p or q)
    
    Returns:
        coefs : ndarray of shape (d, order)
            coefs[b, lag] = coefficient at power `lag` for band b,
            excluding the leading monic coefficient

matrix_companion_eigenvalues(A)
    Compute eigenvalues of block companion matrix for stability checking.
    
    Parameters:
        A : ndarray
            Block companion matrix from build_state_space()
    
    Returns:
        eigenvalues : ndarray
            Eigenvalues of the companion matrix (should have negative real
            parts for stability)

Implementation Notes
--------------------
- All expansions use numpy.convolve for polynomial multiplication
- Results are cast to real via np.real_if_close to discard numerical imaginary parts
- Padding ensures consistent output length when factor product yields lower order
- Truncation removes higher-order terms if product exceeds desired order
- The module assumes continuous-time parameterization (negative real roots)

Numerical Considerations
------------------------
- Convolution may produce coefficients with tiny imaginary parts due to
  floating-point errors - these are discarded with real_if_close
- For high-order polynomials (p > 10), factor multiplication remains stable
- Eigenvalue computation for large block companion matrices may be expensive
  (use only for diagnostics, not within optimization loops)

Dependencies
------------
- numpy for polynomial convolution, linear algebra, and array operations

See Also
--------
- optimizer_utils.pack_params_jones() - Inverse operation: pack coefficients to factors
- optimizer_utils.unpack_params_jones() - Unpack theta to factors
- statespace.build_state_space() - Uses expanded coefficients to build F, G, H

Raises
------
ValueError
    If factor tuple has length not equal to 1 or 2 in expand_jones_to_coeffs()

Usage Example (Internal)
------------------------
Typical usage within parameter transformation pipeline:

    # AR factors for 2-band model with p=2
    ar_factors_band0 = [(0.5,), (0.3, 0.1)]  # linear + quadratic
    ar_factors_band1 = [(0.4,), (0.2, 0.15)]
    ar_factors_list = [ar_factors_band0, ar_factors_band1]
    
    # Expand to coefficient matrix
    A_coefs = build_coefficient_matrices(ar_factors_list, p=2)
    # A_coefs shape: (2, 2) with columns [a₁, a₂] for each band
    
    # Build state-space
    F = build_companion_from_coefs(A_coefs)  # (2p, 2p) block companion
    
    # Check stability
    eigvals = matrix_companion_eigenvalues(F)
    assert np.all(np.real(eigvals) < 0)  # All negative real parts

See optimizer_utils for the complementary packing/unpacking functions that
convert between flat theta vectors and factor representations.
"""

import numpy as np


# ---------------- Positive mapping ----------------

def positive_params(raw):
    """Map unconstrained parameters to strictly positive values."""
    return np.exp(raw)


# ---------------- Jones factor expansion ----------------

def expand_jones_to_coeffs(factors, order):
    """
    Expand a list of Jones factor tuples (a or b parameters) into
    polynomial coefficients.

    This function always expands in AR convention: each factor is treated
    as monic with the parameter on the low-degree side, producing a monic
    polynomial [1, c_{order-1}, ..., c_0] (highest power first).

    For AR polynomials this is the natural form: a parameters directly
    give the monic factors (z + a_0) and (z^2 + a_1 z + a_2),
    and the output [1, a_{p-1}, ..., a_0] is used directly in build_state_space.

    For MA polynomials the math defines factors as (b_0 z + 1) and
    (b_1 z^2 + b_2 z + 1) with constant term 1. This function still
    expands them in AR convention as (z + b_0) etc., giving a monic
    polynomial highest-power-first. build_state_space reverses the result
    with [::-1] to convert to the MA convention [1, b_1, ..., b_q]
    (lowest power first, constant term 1). Do NOT call this function on
    MA factors for diagnostics (e.g. root-finding) without that same reversal.

    Parameters
    ----------
    factors : list of tuples of a (AR) or b (MA) values
        linear factor    -> (param,)         treated as (z + param)
        quadratic factor -> (param1, param2) treated as (z^2 + param1 z + param2)
    order : int
        Expected polynomial order (p for AR, q for MA).

    Returns
    -------
    coeffs : np.ndarray of length order+1
        [1, c_{order-1}, ..., c_0]  (highest power first, monic)
    """
    poly = np.array([1.0])
    for f in factors:
        if len(f) == 1:
            poly = np.convolve(poly, [1.0, f[0]])
        elif len(f) == 2:
            poly = np.convolve(poly, [1.0, f[0], f[1]])
        else:
            raise ValueError(
                f"Factor tuple must have length 1 or 2, got {len(f)}")

    target_len = order + 1
    if len(poly) < target_len:
        poly = np.pad(poly, (target_len - len(poly), 0))
    else:
        poly = poly[-target_len:]

    return np.real_if_close(poly)


# ---------------- Coefficient matrices for all bands ----------------

def build_coefficient_matrices(factors_list, order):
    """
    Expand factor lists for all bands into a (d, order) coefficient matrix.

    Parameters
    ----------
    factors_list : list of length d
        factors_list[b] is the factor list for band b.
    order : int
        Polynomial order (p or q).

    Returns
    -------
    coefs : np.ndarray of shape (d, order)
        coefs[b, lag] = coefficient at power `lag` for band b,
        excluding the leading monic coefficient.
    """
    d = len(factors_list)
    coefs = np.zeros((d, order))
    for b, factors in enumerate(factors_list):
        coeffs = expand_jones_to_coeffs(factors, order)
        # coeffs has length order+1; drop leading 1 (monic)
        coefs[b, :] = coeffs[1:]
    return coefs


# ---------------- Companion eigenvalues (diagnostic) ----------------

def matrix_companion_eigenvalues(A):
    """Compute eigenvalues of block companion matrix."""
    return np.linalg.eigvals(A)