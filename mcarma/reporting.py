"""Map fitted parameters back to the quantities a reader wants to see.

The optimizer works in an unconstrained vector `theta` whose covariance block
is a log-Cholesky packing: off-diagonal entries of L are stored directly, while
each diagonal entry is stored as `log(L_ii**2)`, i.e. the log variance. Nothing
outside the optimizer wants to read that. These helpers undo the packing and
carry standard errors across with it:

  * `chol_theta_to_sigma`  -- packed theta block   -> (L, Sigma)
  * `sigma_to_var_corr`    -- Sigma                -> (variances, correlations)
  * `chol_jacobian`        -- d Sigma / d theta, for the delta method
  * `propagate_sigma_se`   -- cov(theta) block     -> SEs on Sigma entries

The log-variance convention is the one place this package has been bitten
before: a factor of 2 between `theta_ii` and `2*log L_ii` made the Sigma soft
box bind about 100x too high and collapsed multivariate coverage. Anything that
reads or writes the Cholesky block should go through this module rather than
re-deriving the packing.
"""

import numpy as np


def chol_theta_to_sigma(theta_chol, d):
    """Recover L and Sigma from packed theta_chol (log-variance convention)."""
    L = np.zeros((d, d))
    rows, cols = np.tril_indices(d)
    n_chol = d * (d + 1) // 2
    for k in range(n_chol):
        i, j = int(rows[k]), int(cols[k])
        L[i, j] = np.exp(0.5 * theta_chol[k]) if i == j else theta_chol[k]
    Sigma = L @ L.T
    return L, Sigma


def sigma_to_var_corr(Sigma):
    """Decompose Sigma into variances and correlation matrix."""
    var  = np.diag(Sigma)
    std  = np.sqrt(var)
    corr = Sigma / np.outer(std, std)
    return var, corr


def chol_jacobian(theta_chol, d):
    """
    Jacobian J of shape (n_chol, n_chol) where
        J[a, k] = d(Sigma_ij) / d(theta_k)
    with (i,j) and (k -> mn) both indexed by tril_indices(d).

    Uses the log-variance convention: theta_k = log(L_mm^2) for diagonal entries.
    d(Sigma_ij)/d(L_mn) = L_jn * 1_{i=m} + L_in * 1_{j=m}  (lower-tri only)
    d(Sigma_ij)/d(theta_k) = d(Sigma_ij)/d(L_mm) * 0.5 * L_mm
    """
    L, Sigma = chol_theta_to_sigma(theta_chol, d)
    rows, cols = np.tril_indices(d)
    n_chol = len(rows)

    J = np.zeros((n_chol, n_chol))
    for a in range(n_chol):
        i, j = int(rows[a]), int(cols[a])   # Sigma entry (i,j), i>=j
        for k in range(n_chol):
            m, n = int(rows[k]), int(cols[k])   # theta entry -> L[m,n]
            # d(Sigma_ij)/d(L_mn):
            # Sigma_ij = sum_r L_ir L_jr  (r <= min(i,j))
            # nonzero iff (m==i and n<=i) or (m==j and n<=j)
            dS = 0.0
            if m == i and n <= i:
                dS += L[j, n]
            if m == j and n <= j:
                dS += L[i, n]
            # chain rule for log-variance diagonal
            if m == n:   # diagonal entry: theta = log(L_mm^2)
                dS *= 0.5 * L[m, m]
            J[a, k] = dS
    return J


def propagate_sigma_se(theta_chol, cov_theta_chol, d):
    """
    Delta-method SEs for unique Sigma entries given Hessian-based
    covariance of theta_chol.

    Returns
    -------
    Sigma      : (d, d)
    se_sigma   : (d, d) symmetric, SE of each Sigma entry
    var_sigma  : variance
    se_var     : SE of per-band variances (diagonal of Sigma)
    corr       : correlation matrix
    """
    J = chol_jacobian(theta_chol, d)
    cov_sigma_vec = J @ cov_theta_chol @ J.T   # (n_chol, n_chol)

    _, Sigma = chol_theta_to_sigma(theta_chol, d)
    rows, cols = np.tril_indices(d)

    se_sigma = np.zeros((d, d))
    for a in range(len(rows)):
        i, j = int(rows[a]), int(cols[a])
        se_sigma[i, j] = np.sqrt(max(cov_sigma_vec[a, a], 0.0))
        se_sigma[j, i] = se_sigma[i, j]

    var_sigma, corr = sigma_to_var_corr(Sigma)
    # SE of variances = SE of diagonal Sigma entries
    diag_idx = [k for k, (i, j) in enumerate(zip(rows, cols)) if i == j]
    se_var = np.array([se_sigma[int(rows[k]), int(rows[k])] for k in diag_idx])

    return Sigma, se_sigma, var_sigma, se_var, corr